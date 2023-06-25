import torch
import torch.nn as nn
import torchvision.models as models


class Resnet50MetricLearning(nn.Module):
    def __init__(self, identities_num):
        model = models.resnet50(weights= models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, identities_num)  # Substitui a última camada pela nova camada linear
        super(Resnet50MetricLearning, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])  # Camadas do modelo, exceto a última
        self.fc = model.fc  # Última camada do modelo

    def forward(self, x):
        features = self.features(x)
        output = self.fc(features.view(features.size(0), -1))
        return features, output
    
