""" stacked_lstm.py

    Objects to manage Stacked BiLSTM layers  (A Bi-directional stacked LSTM and the underlying LSTMs it employs)

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
from split_layer import SplitLayer
from embedder import Embedder

################################################################################
# LAYERS

class StackedBiLSTM(nn.Module):
    """
    Bi-directional Stacked LSTM, designed to read a whole line and make predictions on each interstitial.  The stacking is done
    simply by using two LSTMs where there was previously one.  They feed into each other before producing output.

    For a line containing N characters, N-1 predictions will be made.

    This layer also commands the "underlying" layers: left2right (A and B), right2left, splitLayer.
    This layer takes care of setting modes, and saving / loading.

    """
    def __init__(self, vec_dim, dim, options={}):
        """
        Instantiate the Layer (which will later be used to connect tensors)

        Parameters
        ----------
        vec_dim : int (width of the vector generated by the vectorizer)

        dim : int (width of vectors in the rest of the network)
        """
        super(StackedBiLSTM, self).__init__()
        
        # Embedder (mostly to upscale the incoming character vectors)
        self.embedder     = Embedder(vec_dim, dim)
        
        # Left to right Stacked LSTM
        self.left2rightA  = LSTMCell(dim, options=options)            # Outputs two length-dim vectors
        self.left2rightB  = LSTMCell(dim, options=options)            # Outputs two length-dim vectors
        
        # Right to left Stacked LSTM
        self.right2leftA  = LSTMCell(dim, options=options)            # Outputs two length-dim vectors
        self.right2leftB  = LSTMCell(dim, options=options)            # Outputs two length-dim vectors

        # Output decision layer
        self.splitLayer  = SplitLayer(dim, options=options)      # Outputs a one-dimensional length-2 vector
        
        self.Lzeros      = pu.var_zeros(dim, ttype=options.get('ttype'))    # Zeros for initialization from left
        self.Lhzeros     = pu.var_zeros(dim, ttype=options.get('ttype'))    # Zeros for initialization (hidden layer)
        self.Rzeros      = pu.var_zeros(dim, ttype=options.get('ttype'))    # same, for the right
        self.Rhzeros     = pu.var_zeros(dim, ttype=options.get('ttype'))
        self.training    = False   # By default, but set to True during training with 'train()'

        
    def get_parameters(self):
        """
        Return a list of trainable parameters for the PyTorch optimizer
        """
        parameters = list(self.parameters())
        parameters.extend( list( self.embedder.parameters() ))
        parameters.extend( list( self.left2rightA.parameters() ))
        parameters.extend( list( self.left2rightB.parameters() ))
        parameters.extend( list( self.right2leftA.parameters() ))
        parameters.extend( list( self.right2leftB.parameters() ))
        parameters.extend( list( self.splitLayer.parameters() ))

        return parameters


    def training_mode(self):
        """
        Set models for training mode
        """
        self.train()
        self.embedder.train()
        self.left2rightA.train()
        self.left2rightB.train()
        self.right2leftA.train()
        self.right2leftB.train()
        self.splitLayer.train()


    def eval_mode(self):
        """
        Set models for evaluation mode, which has, for instance, no dropout
        """
        self.eval()
        self.embedder.eval()
        self.left2rightA.eval()
        self.left2rightB.eval()
        self.right2leftA.eval()
        self.right2leftB.eval()
        self.splitLayer.eval()

        
    def save(self, dirpath):
        """
        Save the current state of the model
        """
        try:
            torch.save(self.state_dict(),  dirpath+'/StackedBiLSTM.pth')
        except:
            raise
        self.embedder.save(dirpath)
        self.left2rightA.save(dirpath, '/left2rightA.pth')
        self.left2rightB.save(dirpath, '/left2rightB.pth')
        self.right2leftA.save(dirpath, '/right2leftA.pth')
        self.right2leftB.save(dirpath, '/right2leftB.pth')
        self.splitLayer.save(dirpath)


    def load(self, dirpath):
        """
        Load the models from a specified directory.  Toss exceptions because models won't exist on the first run.
        """
        try:
            state_dict = torch.load(dirpath+'/StackedBiLSTM.pth', map_location=lambda storage, loc: storage)
        except:
            pass
        self.embedder.load(dirpath)
        self.left2rightA.load(dirpath, '/left2rightA.pth')
        self.left2rightB.load(dirpath, '/left2rightB.pth')
        self.right2leftA.load(dirpath, '/right2leftA.pth')
        self.right2leftB.load(dirpath, '/right2leftB.pth')
        self.splitLayer.load(dirpath)


    def forward(self, line):
        """
        The 'forward()' for this network.  This function takes one sample at a time, where each sample is a line of text.

        Each LSTM, as it proceeds character by character, it makes a prediction *after* having consumed the character on both sides of an interstice.

        Parameters
        ----------
        line : (tensor) list of N vectors (where each vector represents one character)

        Returns
        -------
        (tensor) list of N-1 pairs (each pair of floats represents one interstice-- both the probability of no split and yes split)
        """
        inputs = []    # List of list of tensor.  Will hold the cell and hidden states at each possible split location
        # i.e. the value associated with the first interstice will be at index 0, etc.
        
        ###  Run line through the left2right LSTM  ###
        LCA  = self.Lzeros     # left2right cell state
        LHA  = self.Lhzeros    # left2right hidden state
        LCB  = self.Lzeros     # same for right2left ...
        LHB  = self.Lhzeros
        num = 0
        loc = 0   # First interstice is number 0
        
        for x in line:
            x = self.embedder(x)
        
            # Get the states for interstice number 'loc'
            LCA, LHA = self.left2rightA(x, LCA, LHA)      # Run this character through the Left->-Right LSTM
            LCB, LHB = self.left2rightB(LHA, LCB, LHB)    # Run this character through the second Left->-Right LSTM

            # If we have consumed at least two characters, we will store our first output, which applies to the interstice between the last two characters
            if num >= 1:
                inputs.append( [LCB, LHB] )
                assert(inputs[loc] == [LCB, LHB])
                assert(len(inputs[loc]) == 2)
                loc += 1                                  # For instance, loc=0 represents interstitial between the first and second char
            num += 1
            
        ###  Run line through the right2left LSTM  ###
        RCA  = self.Rzeros
        RHA  = self.Rhzeros
        RCB  = self.Rzeros
        RHB  = self.Rhzeros
        num = 0
        
        for x in reversed(line):                          # Iterate backwards through the same list of vectors
            x = self.embedder(x)
            
            # Get the states for interstice number 'loc'
            RCA, RHA = self.right2leftA(x, RCA, RHA)      # Run this character through the Right->-Left LSTM
            RCB, RHB = self.right2leftB(RHA, RCB, RHB)    # Run this character through the second Right->-Left LSTM
            if num >= 1:
                loc -= 1                                  # Keep this process in reverse alignment with the other
                try:
                    inputs[loc].extend([RCB, RHB])
                    assert(len(inputs[loc]) == 4)
                except:
                    raise
            num += 1   # Just to keep track of how many characters we've consumed
            
        ###  Combine output from both LSTMs and run through the SplitLayer  ###
        #   'preds' will have one fewer element than 'line' because each prediction applies to the interstitial between two elements of 'line'
        preds = None
        for item in inputs:
            LC, LH, RC, RH = item               # Just the output from the second layer of each stacking
            pred = self.splitLayer(LH, RH)      # Splitter layer takes as input the hidden state from both directions
            pred = torch.unsqueeze(pred, 0)     # Add a dimension so that 'tensor_cat()' will work correctly
            if preds is None:
                preds = pred
            else:
                preds = pu.tensor_cat(preds,  pred, 0)   # Build a stack of tensors
                
        try:
            assert(len(line) - 1 == len(preds))
        except:
            err([preds, len(line), len(preds)])
            raise

        return preds
                
                
#######################
class LSTMCell(nn.Module):
    """
    Basic LSTMCell

    Variable scheme
    ---------------
    X : the current input vector

    H : the hidden state

    U : matrices applied to input vectors

    W : matrices applied to the hidden state

    """
    def __init__(self, dim, options={}):
        """
        Instantiate the Layer (which will later be used to connect tensors)
        """
        super(LSTMCell, self).__init__()
        self.activ     = torch.nn.LeakyReLU(negative_slope=0.001, inplace=False)
        self.fc1       = nn.Linear(dim * 2, dim).double()   # starts at width*2 because it has two input vectors: LC, RC
        self.fc2       = nn.Linear(dim, dim).double()
        
        self.Uf  = nn.Linear(dim, dim).double()
        self.Uc  = nn.Linear(dim, dim).double()
        self.Ui  = nn.Linear(dim, dim).double()
        self.Uo  = nn.Linear(dim, dim).double()
        self.Wf  = nn.Linear(dim, dim).double()
        self.Wc  = nn.Linear(dim, dim).double()
        self.Wi  = nn.Linear(dim, dim).double()
        self.Wo  = nn.Linear(dim, dim).double()
        self.training  = False   # By default, but set to True during training with 'train()'
        
        
    def save(self, dirpath, name='LSTMCell'):
        """
        Save the model weights to disk
        """
        torch.save(self.state_dict(), dirpath+'/%s.pth'% name)
        
        
    def load(self, dirpath, name='LSTMCell'):
        """
        Load model weights from disk
        """
        try:
            state_dict = torch.load(dirpath+'/%s.pth'% name, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict)
        except:
            pass

        
    def forward(self, X, C0, H0):
        """
        the 'forward()' for this network.  The mathematics of the gates will follow this scheme, where C0 and H0 are the values coming
        in from previous iterations of this cell and are used to compute H1 (the next hidden layer) and C1, C2 (cell state tensors both
        within this iteration -- sometimes called Ct~ and Ct).

        Parameters
        ----------
        X : Tensor, the current input vector

        C0 : Tensor, the previous cell state

        H0 : Tensor, the previous hidden state
        """
        # Step 1 (Forget Gate): F = sigma(X * Uf + H0 * Wf)
        F = self.Uf(X) + self.Wf(H0)
        F = torch.sigmoid(F)

        # Step 2 (for updating C): C1 = tanh(X * Uc + H0 * Wc)
        C1 = self.Uc(X) + self.Wc(H0)
        C1 = torch.tanh(C1)
        
        # Step 3 (for updating C): I = sigma(X * Ui + H0 * Wi)
        I = self.Ui(X) + self.Wi(H0)
        I = torch.sigmoid(I)
        
        # Step 4 (output gate): O = sigma(X * Uo + H * Wo)
        O = self.Uo(X) + self.Wo(H0)
        O = torch.sigmoid(O)
            
        # Step 5: C2 = F * C0 + I * C1
        C2 = F * C0 + I * C1
        
        # Step 6: H = O * tanh(C)
        H1 = O * torch.tanh(C2)
        
        return C2, H1

    
################################################################################
################################################################################
