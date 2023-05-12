from transformers import AutoTokenizer, AutoModel,  get_scheduler
from torch.utils.data import DataLoader
import torch
from scipy.spatial.distance import cosine, euclidean, cdist
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from src.dataset import Dataset
from src.model import NeuralNetwork


class BB4NormTrainer(object):
	def __init__(self, args, train_dataset=None, ref_dataset=None, test_dataset=None):
		self.args = args
		self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
		
		# Load models
		self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
		self.model = AutoModel.from_pretrained(args.model_name).to(self.device)
		self.basenorm = NeuralNetwork(args.embbed_size).to(self.device)

		# Load datasets (models are needed to make data loaders)
		X_train, y_train = self.mk_set(train_dataset, ref_dataset, self.tokenizer)
		X_test, y_test = self.mk_set(test_dataset, ref_dataset, self.tokenizer)
		self.ref_dataset = ref_dataset
		self.test_dataset = test_dataset

		train_set = Dataset(X_train, y_train)
		test_set = Dataset(X_test, y_test)

		self.train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
		self.test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
		self.num_training_steps = args.epochs * len(self.train_dataloader)

	def train(self):
		optimizer = torch.optim.AdamW(self.basenorm.parameters(), lr=self.args.learning_rate)

		loss_fn = cos_dist
		lr_scheduler = get_scheduler(
			name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
		)

		#Training loop
		self.basenorm.train()
		
		# loop over using tqdm process bar, put loss at the end of progress bar
		tepoch = tqdm(range(self.args.epochs))
		for epoch in tepoch:
			tepoch.set_description(f"Epoch {epoch}")
			for X, y in self.train_dataloader: # both X and y contains n=batch_size mentions and labels embeddings respectively
				batch_loss = None
				for mention, label in zip(X, y):
					pred = self.basenorm(mention) # Single linear layer
					ground_truth = self.basenorm(label)
					loss = loss_fn(pred, ground_truth) # Cosine similarity between embedding of mention and associated label.
					if batch_loss == None:
						batch_loss = loss.reshape(1,1)
					else:
						batch_loss = torch.cat((batch_loss, loss.reshape(1,1)), dim=1) # Appends current loss to all losses in batch

				# Backpropagation
				batch_loss = torch.mean(batch_loss) # Averages loss over the whole batch.
				batch_loss.backward()
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
			tepoch.set_postfix(loss = batch_loss.item())

	def inference(self):
		print("Embedding ontology concept labels...")

		######
		# Build labels/tags embeddings from ontology:
		######
		dd_ref = self.ref_dataset
		dd_test = self.test_dataset

		nbLabtags = 0
		dd_conceptVectors = dict()
		embbed_size = None
		with torch.no_grad():
			for cui in tqdm(dd_ref.keys(), desc='Building embeddings from ontology labels'):
				dd_conceptVectors[cui] = dict()
				dd_conceptVectors[cui][dd_ref[cui]["label"]] = self.basenorm(self.model(self.tokenize(dd_ref[cui]['label']))[0][:,0]).cpu().detach().numpy()# The last hidden-state is the first element of the output tuple
				nbLabtags += 1
				if embbed_size == None:
					embbed_size = len(dd_conceptVectors[cui][dd_ref[cui]["label"]][0])
				if dd_ref[cui]["tags"]:
					for tag in dd_ref[cui]["tags"]:
						nbLabtags += 1
						dd_conceptVectors[cui][tag] = self.basenorm(self.model(self.tokenize(tag))[0][:,0]).cpu().detach().numpy()

		print("Number of concepts in ontology:", len(dd_ref.keys()))
		print("Number of labels in ontology:", nbLabtags)
		print("Done.\n")

		######
		# Build mention embeddings from testing set:
		######
		X_pred = np.zeros((len(dd_test.keys()), embbed_size))
		with torch.no_grad():
			for i, id in tqdm(enumerate(dd_test.keys()), desc ='Building embeddings from test labels'):
				tokenized_mention = torch.tensor(self.tokenize(dd_test[id]['mention']).to(self.device))
				X_pred[i] = self.basenorm(self.model(tokenized_mention)[0][:,0]).cpu().detach().numpy()

		######
		# Nearest neighbours calculation:
		######
		dd_predictions = dict()
		for id in dd_test.keys():
			dd_predictions[id] = dict()
			dd_predictions[id]["pred_cui"] = []

		labtagsVectorMatrix = np.zeros((nbLabtags, embbed_size))
		i = 0
		for cui in dd_conceptVectors.keys():
			for labtag in dd_conceptVectors[cui].keys():
				labtagsVectorMatrix[i] = dd_conceptVectors[cui][labtag]
				i += 1

		print('\tDistance matrix calculation...')
		scoreMatrix = cdist(X_pred, labtagsVectorMatrix, 'cosine')  # cdist() is an optimized algo to distance calculation.
		print("\tDone.")

		# For each mention, find back the nearest label/tag vector, then attribute the associated concept:
		i=0
		for i, id in enumerate(dd_test.keys()):
			minScore = min(scoreMatrix[i])
			j = -1
			stopSearch = False
			for cui in dd_conceptVectors.keys():
				if stopSearch == True:
					break
				for labtag in dd_conceptVectors[cui].keys():
					j += 1
					if scoreMatrix[i][j] == minScore:
						dd_predictions[id]["pred_cui"] = [cui]
						stopSearch = True
						break
		del dd_conceptVectors
		return dd_predictions

	def tokenize(self, sentence):
		return self.tokenizer.encode(sentence, padding="max_length", max_length=self.args.max_length, 
										truncation=True, add_special_tokens = True, return_tensors="pt").to(self.device) # Tokenize input into ids.

	def mk_set(self, dataset, concept_dict, tokenizer):
		# Constructs two dictionnaries containing embeddings of mentions (X) and associated labels (Y) respectively.
		X = dict()
		y = dict()
		with torch.no_grad():
			for i, id in enumerate(dataset.keys()):
				X[i] = self.model(tokenizer.encode(
										dataset[id]['mention'], padding="max_length", max_length=self.args.max_length, 
										truncation=True, add_special_tokens = True, return_tensors="pt").to(self.device))[0][:,0]
				y[i] = self.model(tokenizer.encode(
										concept_dict[dataset[id]['cui'][0]]['label'], padding="max_length", max_length=self.args.max_length, 
										truncation=True, add_special_tokens = True, return_tensors="pt").to(self.device))[0][:,0]
			nbMentions = len(X.keys())
		print("Number of mentions:", nbMentions)
		return X, y


def cos_dist(t1, t2):
    cos = nn.CosineSimilarity()
    cos_sim = cos(t1, t2)*(-1)
    return cos_sim