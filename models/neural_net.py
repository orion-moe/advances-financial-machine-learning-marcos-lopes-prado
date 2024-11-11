import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], num_classes=3, dropout=0.3):
        super(NeuralNet, self).__init__()
        layers = []
        last_size = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))  # Adicionado BatchNorm para acelerar o treinamento
            layers.append(nn.Dropout(dropout))  # Dropout para regularização
            last_size = hidden
        layers.append(nn.Linear(last_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
