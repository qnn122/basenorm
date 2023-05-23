import torch
import torch.nn as nn
from transformers import AutoModel

class NeuralNetwork(nn.Module):
    def __init__(self, embbed_size):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(embbed_size, embbed_size) # Single linear layer
        torch.nn.init.eye_(self.linear.weight) # Linear layer weights initialization

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        x = self.linear(x)
        return x



class BertWithCustomNNClassifier(nn.Module):
    """
    A pre-trained BERT model with a custom classifier.
    The classifier is a neural network implemented in this class.
    """
    
    def __init__(self, linear_size, model_name):
        super(BertWithCustomNNClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(in_features=768, out_features=linear_size) # TODO: automatically get the size of the last layer of BERT (768) instead of hardcoding it
        torch.nn.init.eye_(self.linear1.weight)

        
    def forward(self, tokens):
        bert_output = self.bert(input_ids=tokens)
        x = torch.nn.functional.normalize(bert_output[0][:,0])
        x = self.linear1(x)
        return x
        
    def freeze_bert(self):
        """
        Freezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad=False
    
    def unfreeze_bert(self):
        """
        Unfreezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        both the wieghts of the custom classifier and of the underlying BERT are modified.
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad=True