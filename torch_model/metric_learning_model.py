import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck

from torch_model.resnet_last_stride import ResNet


class Resnet50MetricLearning(nn.Module):
    def __init__(self, identities_num, inference=False):    
        super(Resnet50MetricLearning, self).__init__()
        
        # instantiate resnet50 with last stride = 1 (i.e. higher spatial resolution)                        
        model = ResNet(Bottleneck, layers= [3, 4, 6, 3])
        # to get this state dict:
        #  import torchvision.models 
        #  model = models.resnet50(weights= models.ResNet50_Weights.DEFAULT)
        #  torch.save(model.state_dict(), 'resnet_original_weights.torch.pt')
        model.load_state_dict(torch.load('resnet_original_weights.torch.pt'))
        
        self.batch_norm = nn.BatchNorm2d(num_features=2048)
        model.fc = nn.Linear(model.fc.in_features, identities_num)  # Substitui a última camada pela nova camada linear
        
        self.features = nn.Sequential(*list(model.children())[:-1])  # Camadas do modelo, exceto a última
        self.fc = model.fc  # Última camada do modelo
        self.inference=inference
        
    def forward(self, x):
        features = self.features(x)
        features_norm = self.batch_norm(features)
        output = self.fc(features_norm.view(features_norm.size(0), -1))
        if not self.inference:
            return features, output
        else:
            return features_norm, output
    
