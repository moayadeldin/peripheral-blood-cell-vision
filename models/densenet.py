import torch
import torch.nn as nn
import torch.nn.functional as F

class adjustedDenseNet(nn.Module):
    

   def __init__(self):

      super(adjustedDenseNet,self).__init__()

      self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

      in_features = self.model.classifier.in_features

      self.model.classifier = nn.Identity()

      self.fc1 = nn.Linear(in_features,1000)
      self.fc2 = nn.Linear(1000,8)

   
   def forward(self, x):

      x = self.model(x)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.softmax(x,dim=1)

      return x
   







        