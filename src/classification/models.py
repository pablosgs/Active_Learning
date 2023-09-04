from torch import nn
from transformers import DistilBertModel

class SBERTModel(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, device, dropout=0.5):
        super(SBERTModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_layer_size, input_layer_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_layer_size, hidden_layer_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_size, output_layer_size),
            nn.Sigmoid(),
        )
        self.model.to(device)

        self.embedder = nn.Sequential(*list(self.model.children())[:-4]) 
        self.embedder.to(device)

    def forward(self, x):
        return self.model(x)
    
    def embeddings(self, x):
        return self.embedder(x)
    

class DisModel(nn.Module):
    def __init__(self, input_layer_size, output_layer_size, device):
        super(DisModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_layer_size, input_layer_size),
            nn.ReLU(),
            nn.Linear(input_layer_size, input_layer_size),
            nn.ReLU(),
            nn.Linear(input_layer_size, output_layer_size),
            nn.Sigmoid(),
        )
        self.model.to(device)

    def forward(self, x):
        return self.model(x)
