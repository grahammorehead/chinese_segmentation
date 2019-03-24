""" vectorizer.py

    Objects to manage the vectorization of (mostly) chinese characters

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
import unicodedata
from gmutils import err
import gmutils as gm
from gmutils import pytorch_utils as pu

from utils import *

################################################################################
# SETTINGS

TRAIN_FILE       = "data/training.txt"
TEST_FILE        = "data/test.txt"
VECTORIZER_FILE  = "vectorizer.pkl"
DIM              = 256   # Width of vectors
        
default = {
    'embedding_matrix' : True
    }

################################################################################
# OBJECTS

class Vectorizer(gm.Object):
    """
    An object to convert (mostly) Chinese characters into vectors
    """
    def __init__(self, dim=DIM, options={}):
        """
        Instantiate the object and set options
        """
        self.set_options(options, default)
        self.all_characters = set([])    # A set for now.  Later it will be a list.
        self.dim  = dim                  # The vector size (needed for the embedding matrix method, if used)
        self.size = None                 # Number of known characters

        # For the original binary vector method
        self.vectors = {}                # dict { char : vector }
        self.characters = {}             # dict { vector (binary string) : char }

        # For the embedding matrix method
        self.vocab = {}


    def pass_one(self, filename):
        """
        Reads through a file simply observing and counting each character.  Can legitimately be used on the file 'training.txt' because this function does not
        consider splits.  In such cases it can be thought of as absorbing something from unlabeled data.
        """
        iterator = generate_blind_iterator(filename)
        for line in iterator:
            for c in line:   # Iterate over characters from a single line
                if c not in self.all_characters:
                    self.all_characters.add(c)


    def pass_two(self):
        """
        Make sense of what has been seen and devise a sparse vectorization.  This first attempt at vectorization is using the one-hot method merely so that each
        unique character presents itself to the network as an equally-disambiguated signal from which the network may learn.  Yes it would be more compact and
        possibly efficient to map this embedding into a lower-dimensional space, but it is not yet clear that optimizing for efficiency in this case would retain
        sufficiently-discernible signals for the network to easily learn.  This fact comes directly from my ignorance on the distribution of chinese characters
        across the possbilities of being in two or three-character words vs single-character words.

        A one-hot (binary) encoding such as this assumes that each word is equidistant.  Such is obviously not true in meaning-space, but meaning-space is not our
        concern at this time.  Here we are only concerned with "splitting-space," as it were.  The set of characters are likely not equidistant in this space either,
        but each layer of the network (see model.py) represents a spatial transformation which is not likely to be distance-preserving.

        A better embedding will be left for future work.  Casting this space to a smaller number of dimensions could clearly help with rare characters.
        """
        self.all_characters = sorted(self.all_characters)       # Sort to maintain a specified order
        self.size   = len(self.all_characters)                  # This will determine the max number and width of the vector

        if self.get('embedding_matrix'):
            self.width  = self.dim
            
            # self.size is the number of words in your train, val and test set
            # self.dim is the dimension of the word vectors
            self.embed_matrix = nn.Embedding(self.size, self.dim)
            if torch.cuda.is_available():
                self.embed_matrix = self.embed_matrix.cuda()

            for i, char in enumerate(self.all_characters):
                self.vocab[char] = i

        else:
            basic_width = length_of_binary_rep(self.size)       # The original length of binary representation
            self.width  = basic_width + 1                       # Add a vector element for "specialness"
            special     = set([])
            for i, char in enumerate(self.all_characters):
                vec = i_to_binary_tensor(i, basic_width)        # Binary representation of where character sits in the sorted list

                # Add an element to represent specialness
                if is_foreign_or_punct(char):
                    vec.append(1)                               # Extra element of the vector represents if the char is foreign or punctuation, which deserves its own
                    special.add(char)                           # For such characters it is worth creating a separate field because they tend to behave differently
                else:
                    vec.append(0)                               # Future Work: subdivide classes further and use as features.

                try:
                    assert(len(vec) == self.width)              # Each vector must have this length
                except:
                    err([vec, len(vec), self.width])
                    raise

                self.vectors[char] = vec                        # Store in dict for quick access
                self.characters[vec_to_str(vec)] = char
            print("Fitted Vectorizer understands %d characters, %d of which are special characters (foreign or special)."% (self.size, len(special)))
                
        print("\tFinal vector width:", self.width)
        

    def vectorize(self, char):
        """
        Return the desired vector
        """
        if self.get('embedding_matrix'):
            index = pu.torchvar(self.vocab[char], ttype=torch.LongTensor)
            vec = self.embed_matrix(index).double()
            vec = torch.squeeze(vec)
            # pu.print_info(vec)
            return vec

        else:
            vec = self.vectors.get(char)
            return vec


    def decode(self, vec):
        """
        Decode a vector back into the character it represents
        """
        return self.characters[vec_to_str(vec)]
        
        
    def vectorize_sample(self, sample):
        """
        Vectorizes a data sample, completely ignoring label info (the splits)

        Parameters
        ----------
        sample : list of str (Some are individual chars, some are multiple)

        Returns
        -------
        list of vectors (each one representing a single character)
        """
        output = []
        line = ''.join(sample)   # Collapse into a single line of characters (LABELS DISCARDED)
        
        # Iterate over individual characters (All interstice information is gone)
        for char in line:
            vec = self.vectorize(char)
            output.append(vec)
        return output
        
        
################################################################################
# FUNCTIONS

def length_of_binary_rep(i, verbose=False):
    """
    For some 's' which represents a binary number, find its length
    """
    s = str(bin(i))
    if verbose:  err([i, s])
    s = re.sub(r'^0b0*', '', s)
    if verbose:  err([s, len(s)])
    return len(s)

    
def i_to_binary_tensor(i, width, verbose=False):
    """
    Convert an ordinal int i to a binary vector for encoding.  Used to encode the ordinal number of a character.

    Fixed length of 'width' binary digits means a max of 2**width = 64   (e.g. if 'width=6', the range is 0 to 63)
    """
    output = []
    s = str(bin(i))
    if verbose:  err([i, width, s])
        
    s = re.sub(r'^0b', '0'*width, s)
    S = list(s)[-width:]
    S = map(float, S)
    S = list(S)
    
    if verbose:  err([S, len(S)])
    return S


def is_foreign_or_punct(c):
    """
    Boolean: if 'c' is punctuation or a foreign character expressed in Unicode.  This will be a feature in training.
    """
    if 65281 <= ord(c) <= 65381:
        return True
    elif 12289 <= ord(c) <= 12305:
        return True
    return False


################################################################################
# MAIN

if __name__ == '__main__':
    parser = gm.argparser_ml({'desc': "Vectorizer Tool: vectorizer.py"})
    parser.add_argument('--test_tensor', help='Test i_to_binary_tensor()', required=False, type=int)
    parser.add_argument('--test_binary_length', help='Test length_of_binary_rep()', required=False, type=int)
    args = parser.parse_args()   # Get inputs and options

    if args.train:
        vectorizer = Vectorizer()
        print("Reading data files ...")
        vectorizer.pass_one(TRAIN_FILE)
        vectorizer.pass_one(TEST_FILE)
        vectorizer.pass_two()
        gm.serialize(vectorizer, VECTORIZER_FILE)

    elif args.test:
        vectorizer = gm.deserialize(VECTORIZER_FILE)
        print(vectorizer.vectorize("花"))
        print(vectorizer.vectorize("人"))
        print(vectorizer.vectorize("小"))
        print(vectorizer.vectorize("台"))
        print(vectorizer.vectorize("，"))
        print(vectorizer.vectorize("８"))
        print(vectorizer.vectorize("。"))

    elif args.test_tensor:
        print(i_to_binary_tensor(args.test_tensor, 14, verbose=True))
        
    elif args.test_binary_length:
        print(length_of_binary_rep(args.test_binary_length, verbose=True))

    else:
        print(__doc__)
        
################################################################################
################################################################################

