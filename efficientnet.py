import torch
import torch.nn as nn
import torch.nn.functional as F

class adjustedEfficientNet(nn.Module):

      def __init__(self,fc1_input):

         super(adjustedEfficientNet, self).__init__()

         self.fc1_input=fc1_input
         
         self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)

         self.model.fc = nn.Identity()

         self.flatten = nn.Flatten()
         self.fc1 = nn.Linear(self.fc1_input,1000)
         self.fc2 = nn.Linear(1000,8)

      def forward(self, x):

         x = self.model(x)

         x = self.flatten(x)
         x = self.fc1(x)
         x = F.relu(x)
         x = self.fc2(x)
         x = F.softmax(x, dim=1)

         return x



        