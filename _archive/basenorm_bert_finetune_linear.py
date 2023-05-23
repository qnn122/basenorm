#!/bin/python
# SBATCH --mem=64g
# SBATCH --nodes=1
# SBATCH --nodelist=n102
# SBATCH --cpus-per-task=8
# SBATCH --gres=gpu:1


#######################################################################################################
# Imports:
#######################################################################################################

# External dependencies:
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.models import KeyedVectors, Word2Vec
from scipy.spatial.distance import cosine, euclidean, cdist
from pronto import Ontology
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, get_scheduler
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal libraries:
import numpy as np
import copy
import os
from os import listdir, write
from os.path import isfile, join, splitext
from tqdm import tqdm

###################################################
# Ontological tools:
###################################################

def loader_ontobiotope(filePath):
    """
    Description: A loader of OBO ontology based on Pronto lib.
    (maybe useless...)
    :param filePath: Path to the OBO file.
    :return: an annotation ontology in a dict (format: concept ID (string): {'label': preferred tag,
    'tags': list of other tags, 'parents': list of parents concepts}
    """
    dd_obt = dict()
    onto = Ontology(filePath).terms()
    for o_concept in onto:
        dd_obt[o_concept.id] = dict()
        dd_obt[o_concept.id]["label"] = o_concept.name
        dd_obt[o_concept.id]["tags"] = list()

        for o_tag in o_concept.synonyms:
            dd_obt[o_concept.id]["tags"].append(o_tag.description)

        dd_obt[o_concept.id]["parents"] = list()
        parents = o_concept.superclasses(with_self=False, distance=1).to_set()
        for o_parent in parents:
            dd_obt[o_concept.id]["parents"].append(o_parent.id)

    return dd_obt


def is_desc(dd_ref, cui, cuiParent):
    """
    Description: A function to get if a concept is a descendant of another concept.
    Here, only used to select a clean subpart of an existing ontology (see select_subpart_hierarchy method).
    """
    result = False
    if "parents" in dd_ref[cui].keys():
        if len(dd_ref[cui]["parents"]) > 0:
            if cuiParent in dd_ref[cui]["parents"]:  # Not working if infinite is_a loop (normally never the case!)
                result = True
            else:
                for parentCui in dd_ref[cui]["parents"]:
                    result = is_desc(dd_ref, parentCui, cuiParent)
                    if result:
                        break
    return result


def select_subpart_hierarchy(dd_ref, newRootCui):
    """
    Description: By picking a single concept in an ontology, create a new sub ontology with this concept as root.
    Here, only used to select the habitat subpart of the Ontobiotope ontology.
    """
    dd_subpart = dict()
    dd_subpart[newRootCui] = copy.deepcopy(dd_ref[newRootCui])
    dd_subpart[newRootCui]["parents"] = []

    for cui in dd_ref.keys():
        if is_desc(dd_ref, cui, newRootCui):
            dd_subpart[cui] = copy.deepcopy(dd_ref[cui])

    # Clear concept-parents which are not in the descendants of the new root:
    for cui in dd_subpart.keys():
        dd_subpart[cui]["parents"] = list()
        for parentCui in dd_ref[cui]["parents"]:
            if is_desc(dd_ref, parentCui, newRootCui) or parentCui == newRootCui:
                dd_subpart[cui]["parents"].append(parentCui)

    return dd_subpart


###################################################
# BB4 normalization dataset loader:
###################################################

def loader_one_bb4_fold(l_repPath):
    """
    Description: Load BB4 data from files.
    WARNING: OK only if A1 file is read before its A2 file (normally the case).
    :param l_repPath: A list of directory path containing set of A1 (and possibly A2) files.
    :return:
    """
    ddd_data = dict()

    i = 0
    for repPath in l_repPath:

        for fileName in listdir(repPath):
            filePath = join(repPath, fileName)

            if isfile(filePath):

                fileNameWithoutExt, ext = splitext(fileName)
                
                if ext == ".a1":

                    with open(filePath, encoding="utf8") as file:
                    
                        if fileNameWithoutExt not in ddd_data.keys():

                            ddd_data[fileNameWithoutExt] = dict()
                            for line in file:

                                l_line = line.split('\t')

                                if l_line[1].split(' ')[0] == "Title" or l_line[1].split(' ')[0] == "Paragraph":
                                    pass
                                else:
                                    exampleId = "bb4_" + "{number:06}".format(number=i)

                                    ddd_data[fileNameWithoutExt][exampleId] = dict()

                                    ddd_data[fileNameWithoutExt][exampleId]["T"] = l_line[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["type"] = l_line[1].split(' ')[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["mention"] = l_line[2].rstrip()

                                    if "cui" not in ddd_data[fileNameWithoutExt][exampleId].keys():
                                        ddd_data[fileNameWithoutExt][exampleId]["cui"] = list()
                                    i += 1

        for fileName in listdir(repPath):
            filePath = join(repPath, fileName)

            if isfile(filePath):

                fileNameWithoutExt, ext = splitext(fileName)
                
                if ext == ".a2":

                    with open(filePath, encoding="utf8") as file:

                        if fileNameWithoutExt in ddd_data.keys():

                            for line in file:
                                l_line = line.split('\t')

                                l_info = l_line[1].split(' ')
                                Tvalue = l_info[1].split(':')[1]

                                for id in ddd_data[fileNameWithoutExt].keys():
                                    if ddd_data[fileNameWithoutExt][id]["T"] == Tvalue:
                                        if ddd_data[fileNameWithoutExt][id]["type"] == "Habitat" or \
                                                ddd_data[fileNameWithoutExt][id]["type"] == "Phenotype":
                                            cui = "OBT:" + l_info[2].split('Referent:')[1].rstrip().replace('OBT:', '')
                                            ddd_data[fileNameWithoutExt][id]["cui"].append(cui)
                                        elif ddd_data[fileNameWithoutExt][id]["type"] == "Microorganism":
                                            cui = l_info[2].split('Referent:')[1].rstrip()
                                            ddd_data[fileNameWithoutExt][id]["cui"] = [cui]  # No multi-normalization for microorganisms
    return ddd_data


def extract_data(ddd_data, l_type=[]):
    """

    :param ddd_data:
    :param l_type:
    :return:
    """
    dd_data = dict()

    for fileName in ddd_data.keys():
        for id in ddd_data[fileName].keys():
            if ddd_data[fileName][id]["type"] in l_type:
                dd_data[id] = copy.deepcopy(ddd_data[fileName][id])
    return dd_data


###################################################
# An accuracy function:
###################################################

def accuracy(dd_pred, dd_resp):
    totalScore = 0.0

    for id in dd_resp.keys():
        score = 0.0
        l_cuiPred = dd_pred[id]["pred_cui"]
        l_cuiResp = dd_resp[id]["cui"]
        if len(l_cuiPred) > 0:  # If there is at least one prediction
            for cuiPred in l_cuiPred:
                if cuiPred in l_cuiResp:
                    score += 1
            score = score / max(len(l_cuiResp), len(l_cuiPred))  # multi-norm and too many pred

        totalScore += score  # Must be incremented even if no prediction

    totalScore = totalScore / len(dd_resp.keys())

    return totalScore


###################################################
# Preprocessing tools:
###################################################

def lowercaser_mentions(dd_mentions):
    dd_lowercasedMentions = copy.deepcopy(dd_mentions)
    for id in dd_lowercasedMentions.keys():
        dd_lowercasedMentions[id]["mention"] = dd_mentions[id]["mention"].lower()
    return dd_lowercasedMentions


def lowercaser_ref(dd_ref):
    dd_lowercasedRef = copy.deepcopy(dd_ref)
    for cui in dd_ref.keys():
        dd_lowercasedRef[cui]["label"] = dd_ref[cui]["label"].lower()
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(tag.lower())
            dd_lowercasedRef[cui]["tags"] = l_lowercasedTags
    return dd_lowercasedRef


#######################################################################################################
# Classes:
#######################################################################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.mention = X
        self.label = y

    def __getitem__(self, idx):
        mention = self.mention[idx]
        label = self.label[idx]
        sample = (mention, label)
        return sample

    def __len__(self):
        return len(self.mention)

class NeuralNetwork(nn.Module):
    def __init__(self, embbed_size):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(embbed_size, embbed_size) # Single linear layer
        torch.nn.init.eye_(self.linear.weight) # Linear layer weights initialization

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        x = self.linear(x)
        return x

#######################################################################################################
# Functions:
#######################################################################################################

def tokenize(sentence):
    return tokenizer.encode(sentence, padding="max_length", max_length=max_length, truncation=True, add_special_tokens = True, return_tensors="pt").to(device) # Tokenize input into ids.

def mk_set(dataset, concept_dict, tokenizer):
    # Constructs two dictionnaries containing tokenized mentions (X) and associated labels (Y) respectively.
    X = dict()
    y = dict()
    for i, id in enumerate(dataset.keys()):
        X[i] = tokenizer.encode(dataset[id]['mention'], padding="max_length", max_length=max_length, truncation=True, add_special_tokens = True, return_tensors="pt")
        y[i] = tokenizer.encode(concept_dict[dataset[id]['cui'][0]]['label'], padding="max_length", max_length=max_length, truncation=True, add_special_tokens = True, return_tensors="pt")
    nbMentions = len(X.keys())
    print("Number of mentions:", nbMentions)
    return X, y

def inference():
    print("Embedding ontology concept labels...")

    ######
    # Build labels/tags embeddings from ontology:
    ######
    nbLabtags = 0
    dd_conceptVectors = dict()
    embbed_size = None
    with torch.no_grad():
        for cui in tqdm(dd_ref.keys(), desc='Building embeddings from ontology labels'):
            dd_conceptVectors[cui] = dict()
            dd_conceptVectors[cui][dd_ref[cui]["label"]] = basenorm(model(tokenize(dd_ref[cui]['label']))[0][:,0]).cpu().detach().numpy()# The last hidden-state is the first element of the output tuple
            nbLabtags += 1
            if embbed_size == None:
                embbed_size = len(dd_conceptVectors[cui][dd_ref[cui]["label"]][0])
            if dd_ref[cui]["tags"]:
                for tag in dd_ref[cui]["tags"]:
                    nbLabtags += 1
                    dd_conceptVectors[cui][tag] = basenorm(model(tokenize(tag))[0][:,0]).cpu().detach().numpy()
    print("Number of concepts in ontology:", len(dd_ref.keys()))
    print("Number of labels in ontology:", nbLabtags)
    print("Done.\n")

    ######
    # Build mention embeddings from testing set:
    ######
    X_pred = np.zeros((len(dd_test.keys()), embbed_size))
    with torch.no_grad():
        for i, id in tqdm(enumerate(dd_test.keys()), desc ='Building embeddings from test labels'):
            tokenized_mention = torch.tensor(tokenize(dd_test[id]['mention']).to(device))
            X_pred[i] = basenorm(model(tokenized_mention)[0][:,0]).cpu().detach().numpy()

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
    # (doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

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

################################################
print("\nLOADING DATA:\n")
################################################

print("loading OntoBiotope...")
dd_obt = loader_ontobiotope("BB4/OntoBiotope_BioNLP-OST-2019.obo")
print("loaded. (Nb of concepts in OBT =", len(dd_obt.keys()), ")")

print("\nExtracting Bacterial Habitat hierarchy:")
dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')
print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ")")

print("\nLoading BB4 corpora...")
ddd_dataAll = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_train", "BB4/BioNLP-OST-2019_BB-norm_dev",
                                    "BB4/BioNLP-OST-2019_BB-norm_test"])
dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")

ddd_dataTrain = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_train"])
dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat"])  # ["Habitat", "Phenotype", "Microorganism"]
print("loaded.(Nb of mentions in train =", len(dd_habTrain.keys()), ")")

ddd_dataDev = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_dev"])
dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
print("loaded.(Nb of mentions in dev =", len(dd_habDev.keys()), ")")

ddd_dataTrainDev = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_train", "BB4/BioNLP-OST-2019_BB-norm_dev"])
dd_habTrainDev = extract_data(ddd_dataTrainDev, l_type=["Habitat"])
print("loaded.(Nb of mentions in train+dev =", len(dd_habTrainDev.keys()), ")")

ddd_dataTest = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_test"])
dd_habTest = extract_data(ddd_dataTest, l_type=["Habitat"])
print("loaded.(Nb of mentions in test =", len(dd_habTest.keys()), ")")

################################################
print("\n\nPREPROCESSINGS:\n")
################################################

print("Mentions lowercasing...")
dd_BB4habTrain_lowercased = lowercaser_mentions(dd_habTrain)
dd_BB4habDev_lowercased = lowercaser_mentions(dd_habDev)
print("Done.\n")

print("Lowercase references...")
dd_habObt_lowercased = lowercaser_ref(dd_habObt)

dd_ref = dd_habObt_lowercased
dd_train = dd_BB4habTrain_lowercased
dd_test = dd_BB4habDev_lowercased
print("Done.")

################################################
print("\nINITIALIZING\n")
################################################

global device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")

################################################
print("\nLOADING EMBEDDING MODEL:\n")
################################################

global model, tokenizer, max_length, embbed_size
model_name = 'dmis-lab/biobert-base-cased-v1.1'
model = AutoModel.from_pretrained(model_name).to(device)
embbed_size = 768
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 40

X_train, y_train = mk_set(dd_train, dd_ref, tokenizer)
X_test, y_test = mk_set(dd_test, dd_ref, tokenizer)

train_set = Dataset(X_train, y_train)
test_set = Dataset(X_test, y_test)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

basenorm = NeuralNetwork(embbed_size).to(device)

# Training parameters
learning_rate = 1e-5
epochs = 5
#optimizer = torch.optim.NAdam(basenorm.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(basenorm.parameters(), lr=learning_rate)
num_training_steps = epochs * len(train_dataloader)

# Activation function. Calculates a cosine similarity*(-1) between mention and label vectors.
def cos_dist(t1, t2):
    cos = nn.CosineSimilarity()
    cos_sim = cos(t1, t2)*(-1)
    return cos_sim

loss_fn = cos_dist
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

#Training loop
model.train()
basenorm.train()
for epoch in tqdm(range(epochs)):
    for X, y in tqdm(train_dataloader): # both X and y contains n=batch_size tokenized mentions and labels respectively
        batch_loss = None
        for tokenized_mention, tokenized_label in zip(X, y):
            tokenized_mention = tokenized_mention.to(device)
            tokenized_label = tokenized_label.to(device)
            pred = basenorm(model(tokenized_mention)[0][:,0]) # Taking last hidden state of the embedding model and piping it into a linear layer.
            ground_truth = basenorm(model(tokenized_label)[0][:,0])
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
    print(f"Fine-tuning: Epoch n° {epoch}, loss = {batch_loss.item()}")

# Inference
dd_predictions = inference()

print("Evaluating BB4 results on BB4 dev...")
score_BB4_onDev = accuracy(dd_predictions, dd_habDev)
print("score_BB4_onDev:", score_BB4_onDev)