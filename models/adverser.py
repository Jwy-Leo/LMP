import numpy 
import torch
import torch.nn as nn
import pdb
class adverser(nn.Module):
    def __init__(self,inplanes,kernel):
        super(adverser, self).__init__()
        self.conv_1 = nn.Conv2d(inplanes, inplanes*2, kernel,padding=1,bias=True)
        self.conv_2 = nn.Conv2d(inplanes*2, inplanes*2, kernel,padding=1,bias=True)
        self.conv_3 = nn.Conv2d(inplanes*2, 2, kernel,padding=1,bias=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()   
    def forward(self,x):
        h1 = self.conv_1(x)
        h1 = self.relu(h1)

        h2 = self.conv_2(h1)
        h2 = self.relu(h2)

        h3 = self.conv_3(h2) 
        return h3

class critic(nn.Module):
    def __init__(self):
        super(critic,self).__init__()
        self.adv = nn.Sequential( 
            adverser(512,3),
        )
    def forward(self,x):
        score = self.adv(x)   
        return score

