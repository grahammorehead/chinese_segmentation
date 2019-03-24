""" embedder.py

    Objects to manage an embedding layer -- meant to transform character vectors into some other-dimensional space.

"""
import sys
import os, re
import time
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
# SETTINGS

default = {
}

################################################################################
# OBJECTS

class Embedder(nn.Module):
    """
    Layer(s) that predict whether to split here or not based on input from both LSTMs
    """
    def __init__(self, in_dim, dim, options={}):
        """
        Instantiate the Layer (which will later be used to connect tensors)
        """
        super(Embedder, self).__init__()
        self.activ     = torch.nn.LeakyReLU(negative_slope=0.001, inplace=False)
        self.fc_final  = nn.Linear(in_dim, dim).double()
        self.training  = False   # By default, but set to True during training with 'train()'


    def save(self, dirpath):
        """
        Save the model weights to disk
        """
        torch.save(self.state_dict(),  dirpath+'/Embedder.pth')
        
        
    def load(self, dirpath):
        """
        Load model weights from disk
        """
        try:
            state_dict = torch.load(dirpath+'/Embedder.pth', map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict)
        except:
            pass

        
    def forward(self, x):
        """
        the 'forward()' for this network

        Parameters
        ----------
        x : (tensor) binary vector representing a given character

        Returns
        -------
        (tensor) vector in a lower-dimensional space

        """
        x = self.fc_final(x)
        x = self.activ(x)
        
        return x

    
################################################################################
################################################################################
