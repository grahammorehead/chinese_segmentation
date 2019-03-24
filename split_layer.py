""" split_layer.py

    Objects to manage the 'SplitLayer layers  (The final layers that convert LSTM output into predictions)

"""
import sys
import os, re
import time
from numpy.random import choice
import numpy as np
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gmutils import err
import gmutils as gm
from gmutils import pytorch_utils as pu

from utils import *

################################################################################
# LAYERS

class SplitLayer(nn.Module):
    """
    Layer(s) that predict whether to split here or not based on input from both LSTMs
    """
    def __init__(self, dim, options={}):
        """
        Instantiate the Layer (which will later be used to connect tensors)
        """
        super(SplitLayer, self).__init__()
        self.activ     = torch.nn.LeakyReLU(negative_slope=0.001, inplace=False)
        self.fc1       = nn.Linear(dim*2, dim*2).double()   # starts at width*2 because it has two input vectors: LC, RC
        self.fc2       = nn.Linear(dim*2, dim).double()
        self.fc_final  = nn.Linear(dim, 1).double()
        self.training  = False   # By default, but set to True during training with 'train()'

        
    def save(self, dirpath):
        """
        Save the model weights to disk
        """
        torch.save(self.state_dict(),  dirpath+'/SplitLayer.pth')
        
        
    def load(self, dirpath):
        """
        Load model weights from disk
        """
        try:
            state_dict = torch.load(dirpath+'/SplitLayer.pth', map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict)
        except:
            pass

        
    def forward(self, L, R):
        """
        the 'forward()' for this network

        Parameters
        ----------
        LC, RC : Tensors (Cell and Hidden states for Left and Right)
        """
        if pu.has_improper_values(L)  or  pu.has_improper_values(R):
            err([L, R])
            raise(NaN_Exeption)
        x = torch.cat((L, R))
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        x = self.activ(x)
        x = self.fc_final(x)
        x = torch.sigmoid(x)
        
        return x

    
################################################################################
################################################################################
