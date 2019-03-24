""" model.py

    Objects to manage a Chinese word tokenizer model

    A Bi-directional LSTM is used to read through each line and make a prediction on each "interstice" (the possible space between two characters)

"""
import sys
import os, re
import time
import random
import math
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
from vectorizer import Vectorizer
from embedder import Embedder
# from stacked_lstm import StackedBiLSTM
from simple_lstm import SimpleBiLSTM

################################################################################
# SETTINGS

TRAIN_FILE       = "data/training.txt"
TEST_FILE        = "data/test.txt"
VECTORIZER_FILE  = "vectorizer.pkl"
BATCH_SIZE       = 100
DIM              = 256   # Width of vectors in the network (not necessarily the character vectors themselves)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(12345)    # comment-out if not needed

default = {
    'learning_rate' : 0.000002,
    'learning_threshold' : 0.1,
    'steps' : 10,
    'min_steps' : 10,
    'class_weights' : [0.3851238010580797, 0.6148761989419204],   # Apriori observed: [ P(no-split) , P(split) ]
    'ttype' : torch.DoubleTensor,
    'zeros_only' : False,
    'embedding_matrix' : True
}

################################################################################
# Non-Layer OBJECTS

###########################    
class Splitter(gm.Object):
    """
    An object to split (mostly) Chinese characters into words, i.e. tokenization.  This is the main object that interfaces with the rest
    of the code.

    The 'Splitter' is an object that contains several underlying models that work together:
      - The BatchGraph (that holds all the tensors for a given batch)
      - The Simple or Stacked BiLSTM
         - which interfaces with two LSTMs
         - and a 'SplitLayer' (which converts the output of the LSTMs into predictions)

    For each line containing N characters, N-1 predictions will be made. 
    """
    def __init__(self, vec_dim, dim, options={}):
        """
        Instantiate the object and set options
        """
        print("options:", options)
        self.set_options(options, default)
        options['ttype']      = self.get('ttype')
        self.validation_graph = None
        # self.bilstm           = StackedBiLSTM(vec_dim, dim, options=options)         # A Bi-directional LSTM
        self.bilstm           = SimpleBiLSTM(dim, options=options)         # A Bi-directional LSTM
        self.embedder         = Embedder(vec_dim, dim, options=options)         # Embedder (mostly for upscaling incoming vectors)
        self.class_weights    = pu.torchvar(self.get('class_weights'), ttype=self.get('ttype'))

        # Loss functions
        self.coef1          = pu.torchvar([0.1])
        self.coef2          = pu.torchvar([0.2])
        self.coef3          = pu.torchvar([0.3])
        self.coef4          = pu.torchvar([0.4])
        self.accuracyLoss   = pu.AccuracyLoss()
        self.skewLoss       = pu.SkewedL1Loss(self.class_weights)
        self.criterion_CE   = nn.CrossEntropyLoss(weight=self.class_weights, size_average=False)
        self.criterion_MSE  = nn.MSELoss()
        self.orig_loss      = None
        
        if torch.cuda.is_available():
            self.bilstm         = self.bilstm.cuda()
            self.coef1          = self.coef1.cuda()
            self.coef2          = self.coef2.cuda()
            self.coef3          = self.coef3.cuda()
            self.coef4          = self.coef4.cuda()
            self.accuracyLoss   = self.accuracyLoss.cuda()
            self.skewLoss       = self.skewLoss.cuda()
            self.criterion_CE   = self.criterion_CE.cuda()
            self.criterion_MSE  = self.criterion_MSE.cuda()

        self.eval_mode()   # eval mode by default


    def parameters(self):
        """
        Return a list of trainable parameters for the PyTorch optimizer
        """
        return list(self.bilstm.get_parameters())


    def training_mode(self):
        """
        Set models for training mode
        """
        self.bilstm.training_mode()


    def eval_mode(self):
        """
        Set models for evaluation mode, which has, for instance, no dropout
        """
        self.bilstm.eval_mode()
        

    def save(self, epoch, batch_num):
        """
        Save the current state of the model.  The name of the saved file will contain information about:
          - validation loss
          - epoch number
          - batch number

        Parameters
        ----------
        epoch : int
        batch_num : int
        """
        loss    = self.evaluate()
        dirpath = self.get('model_dir') + "/L%0.64f_E%d_B%d"% (loss, epoch, batch_num)
        saved   = False
        while not saved:
            try:
                gm.mkdirs([dirpath])
                self.bilstm.save(dirpath)
                if not self.get('silent'):
                    sys.stderr.write('\tSaved to: ' + dirpath + '\n')
                saved = True
            except:
                raise
        return dirpath


    def load(self, dirpath):
        """
        Load the model from a specified directory
        """
        if not self.get('silent'):
            sys.stderr.write("Loading:" + dirpath + " ...\n")
        self.bilstm.load(dirpath)
        self.orig_loss = pu.loss_from_filename(dirpath)
        
        
    def load_good(self):
        """
        Load a model with a low loss (stochastically).  This function effectuates a Poisson-beam search.
        """
        try:
            good = pu.good_model_file_by_loss(self.get('model_dir'))
            self.load(good)
        except:
            pass   # Before a model has been saved, this would raise an exception


    def load_best(self):
        """
        Load the model with the lowest loss
        """
        try:
            best = pu.best_model_file_by_loss(self.get('model_dir'))
            self.load(best)
        except:
            pass   # Before a model has been saved, this would raise an exception


    def reload(self, options={}):
        """
        Look again to the folder of saved models and load one of them
        """
        self.clear_chaff()        # Get rid of bad/old models
        try:
            if options.get('best'):
                self.load_best()          # Load the best model (Lowest validation loss)
            else:
                self.load_good()          # Load a "good" model (maybe not the best-- effectively a Poisson-beam search)
        except:
            raise   # Raises an exception the first time-- before any model file yet exists

        
    def clear_chaff(self):
        """
        Remove some model files having higher loss.  Keep the 25 best.
        """
        pu.clear_chaff_by_loss(self.get('model_dir'), MAX=10)
        

    def get_validation_set(self, iterator):
        """
        Get and keep a validation set to be used during training.  This is only used to mark each model with a loss number.  It's important that this number
        NOT come from the training data NOR from the test data (because it will be used to select a saved model for further training).
        """
        validation = next(iterator)
        while len(validation) < 1000:
            validation.extend( next(iterator) )
        self.validation_graph = BatchGraph(self.bilstm, self.embedder, validation)
        

    def refresh_optimizer(self, lr):
        """
        Generate the optimizer.  Can be done at the beginning of each step-- allows for dynamic learning rate adjustment
        """
        # self.optimizer  = torch.optim.Adam(self.parameters(), lr=lr, amsgrad=True)
        self.optimizer  = torch.optim.RMSprop(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)


    def evaluate(self):
        """
        Evaluate using the in-house validation set
        """
        self.eval_mode()                                       # Just sets the model for 'eval' mode (no dropouts)
        preds, labels  = self.validation_graph.forward()
        loss           = self.loss(preds, labels)
        
        return loss.item()                                     # Gets just the float value of the loss


    def predict_stats(self, batch):
        """
        Predict splits for one batch.  Return results
        """
        graph = BatchGraph(self.bilstm, self.embedder, batch)
        graph.forward()
        all, tp, tn, fp, fn, labels = graph.stats()
        
        return all, tp, tn, fp, fn, labels

    
    def predict_random(self, N):
        """
        Using merely the known class weights, draw from a random distribution

        Parameters
        ----------
        N : int (number of results to return)

        Returns
        -------
        list of int
        """
        choices = choice([0,1], N, p=self.class_weights)
        return choices.tolist()
        

    def loss(self, preds, labels, options={}):
        """
        Compute a loss function by hand
        """
        loss = 0
        n    = 0
        for i, pred in enumerate(preds):
            label = labels[i]
            n += 1
            L = (1 + torch.abs(label - pred) )**4 - 1
            if n < 5:   # Just print out the first few to get an idea
                print("\tLabel: %d   Pred: %0.5f   Loss: %0.5f"% (label, pred, L))
            loss += L
        loss = loss / n
        
        return loss
    
        
    def fit(self, batch, _monitor, options={}, closure=None, verbose=False):
        """
        Train the classifier on this batch

        Parameters
        ----------
        batch : list of list

        _monitor : dict

        """
        if verbose:  err()

        if self.get('fake'):
            batch = self.fake_batch()
            
        self.training_mode()   # Set all layers to training mode
        epoch       = _monitor.get('epoch')
        start_lr    = self.get('learning_rate')
        start_lt    = self.get('learning_threshold')
        lr          = pu.learning_rate_by_epoch(epoch, start_lr)
        lt          = pu.loss_threshold_by_epoch(epoch, start_lt)   # If loss drops below this, move on to next batch
        grad_max    = 2.0
        grad_norm   = 2
        last_loss   = 999

        self.refresh_optimizer(lr)
        self.optimizer.zero_grad()
        graph = BatchGraph(self.bilstm, self.embedder, batch)   # This object holds all of the tensors and their connections to each other
        
        # Execute a number of steps, stopping if necessary
        for step in range(self.get('steps')):
            if step > self.get('min_steps'):
                if last_loss < lt:
                    break   # IF loss below this threshold, we've learned enough for now, move on (RARE)
 
            # FORWARD
            try:
                preds, labels = graph.forward()
            except NaN_Exeption:
                break
            except:
                raise
            loss          = self.loss(preds, labels)
            
            # BACKWARD
            loss.backward(retain_graph=True)
            self.optimizer.step(closure=closure)
            this_loss = loss.item()
            if this_loss >= last_loss:
                if options.get('adaptive_learning_rate'):
                    lr = 0.95 * lr
                    self.refresh_optimizer(lr)
                if lr < 1e-12:
                    break
            
            if not options.get('silent'):
                if step == 0:  print()
                line_no = _monitor['i']
                print('[e:%d l:%d s:%d] (lr %0.1E  lt %0.1E) loss: %.16f'% (epoch, line_no, step, lr, lt, this_loss))

            last_loss = this_loss

        return last_loss


    def fake_batch(self):
        """
        Generate a fake batch for debugging
        """
        def fake_sample():
            labels = []    # 'labels' will be the 'XNOR' of the current vector with the previous one
            vectors = []   # 'vectors' will be simple 0 or 1
            for i in range(10):

                # Generate a fake vector of all 1's or 0's
                if random.randint(0,1):
                    vectors.append([1] * 14)
                else:
                    vectors.append([0] * 14)

                # Generate labels based on a simple XNOR function
                if i > 0:
                    if vectors[-2][0] == vectors[-1][0]:   # If the last two vectors are the same, then label=1
                        label = 1
                    else:
                        label = 0
                    labels.append(label)

            return  [labels, vectors]
            
        batch = []
        for i in range(BATCH_SIZE):
            batch.append( fake_sample() )
            
        return batch
    

#############################
class BatchGraph(gm.Object):
    """
    An object to hold the tensors created for a single batch and all of the connections between them
    """
    def __init__(self, bilstm, embedder, batch, options={}):
        """
        Instantiate the object, create the tensors, tie them together
        """
        self.set_options(options, default)
        self.bilstm      = bilstm
        self.embedder    = embedder        # Layers to (if needed) bring the dimensionality of the vectors in line with what the network expects
        self.labels      = None            # Will hold all labels from this batch
        self.vector_list = []              # List of list of vectors (one list of vectors for each sample)
        self.preds       = None

        try:   # Collect Labels (Assuming they aren't all 'None')
            # First, Collate a stacked tensor containing all labels from all samples in this batch
            for item in batch:
                labels, vectors = item     # A tuple related to one sample-- (list of N-1 labels, list of N char-vectors)
                if labels is None:
                    raise ValueException("Unlabeled Data")

                labels = pu.torchvar(labels, ttype=torch.LongTensor)   # Convert to tensor (on GPU if available).  'torch.long' because the criterion needs ints
                if self.labels is None:
                    self.labels = labels
                else:
                    self.labels = pu.tensor_cat(self.labels,  labels,  0)   # Build a stack of tensors
        except:
            pass   # In the case of real-world data, the labels will all be 'None'
                    
        # Second, Collate a stacked tensor containing all vectors from all samples in this batch (these will be used to make predictions later)
        for item in batch:
            labels, vectors = item                      # A tuple related to one sample-- (list of N-1 labels, list of N char-vectors)
            # (this time ignoring 'labels' -- Labels were done separately to make this function more easily read by a human)
            vectors = self.torchify(vectors)            # Convert to a list of tensors (on GPU if available)
            if not self.get('embedding_matrix'):
                vectors = self.embed_vectors(vectors)   # If needed, convert using a layer to the right dimensionality
            self.vector_list.append(vectors)


    def embed_vectors(self, vectors):
        """
        Use these layers to (if needed) bring the dimensionality of the vectors in line with what the network expects
        """
        output = []
        for x in vectors:
            x = self.embedder(x)
            output.append(x)
            
        return output
            

    def torchify(self, vectors):
        """
        Convert a list of vectors to torch tensors
        """
        output = []
        for vector in vectors:
            if not torch.is_tensor(vector):
                vector = pu.torchvar(vector, ttype=self.get('ttype'))
            output.append(vector)
        return output


    def forward(self):
        """
        Runs the data forward through the network to generate outputs.

            First, iterate over the input vectors for each sample.

            Each sample comprises a list of N (mostly) Chinese characters that have been vectorized.  These will be processed through a BiLSTM
            to produce an output list of N-1 floats, each representing the likelihood (between 0-1) of a split.

            'N' is not specified because the batch is an inhomogeneous list, i.e. each sample is a different length.
        """
        self.preds = None
        for vectors in self.vector_list:          # Each 'vectors' var holds a list of (tensor) vectors that comprises the vectors from an individual sample
            pred = self.bilstm.forward(vectors)   # Returns the list of N-1 pairs associated with an individual sample
            
            if self.preds is None:
                self.preds = pred
            else:
                self.preds = pu.tensor_cat(self.preds,  pred,  0)   # Build a stack of tensors comprised of all 'pred' from all vectors in this sample
                # The final number of 'preds' is not determined, but it is the same as the number of 'labels'

        try:
            assert(len(self.labels) == len(self.preds))
        except:
            err([len(self.labels), len(self.preds)])
            raise

        return self.preds, self.labels
        

    def stats(self):
        """
        Return some statistics on how well the model performed

        We will use the values (preds and labels) already contained in this object
        """
        # If speed is an issue, use this func instead (needs preproccessing):
        #   F1, TP, TN, FP, FN  = pu.F1(preds, labels)

        # labels = self.labels.data.numpy().tolist()
        labels = self.labels
        preds  = self.preds.cpu().data.numpy()
        ALL    = len(labels)
        TP     = TN = FP = FN = 0

        print()
        for i, y in enumerate(self.labels):
            
            ##  y=label, x=pred  ##
            y = int( y.cpu().numpy() )
            pred = preds[i]
            if pred > 0.5:
                x = 1
            else:
                x = 0

            # Print a few to get an idea ...
            if i < 11:
                diff = abs(y-pred)
                if diff > 0.5:
                    print("Label: %d   Pred: %0.5f   Pred int: %d   Diff: %0.5f"% (y, pred, x, diff))
                else:
                    print("Label: %d   Pred: %0.5f   Pred int: %d"% (y, pred, x))
                
            if y == 1  and  x == 1:
                TP += 1
            elif y == 0  and  x == 0:
                TN += 1
            elif y == 0  and  x == 1:
                FP += 1
            elif y == 1  and  x == 0:
                FN += 1

        return ALL, TP, TN, FP, FN, labels

    
            
################################################################################
# FUNCTIONS

def train_epoch(model, vectorizer, filename, epoch, skip=None, options={}):
    """
    Execute one epoch of training
    """
    _monitor = gm.monitor_setup(filename, options={'skip':skip})   # For monitoring progress
    _monitor['epoch'] = epoch
    iterator = generate_batch_iterator(vectorizer, filename, N=BATCH_SIZE, _monitor=_monitor)    # The data iterator
    
    sys.stderr.write("Compiling validation set ...\n")
    model.get_validation_set(iterator)   # The Validation set will comprise the first 300 samples (Future Work: shuffle the data)
    sys.stderr.write("\tValidation set Done.\n")

    # For the sake of having a consistent validation set, execute skip-ahead only on the batch iterator
    gm.monitor_skip(_monitor, iterator)    
    
    batch_num = 0
    sys.stderr.write("Beginning training iteration ...\n")
    for batch in iterator:
        batch_num += 1
        if model.get('no_reload')  and  batch_num > 1:
            pass
        else:
            model.reload(options={'best':True})                    # Load another model
        model.fit(batch, _monitor, verbose=False, options=options)
        model.save(epoch, batch_num)      # Save the latest model

    
def train_splitter(vectorizer, filename, options={}):
    """
    Instantiates a Splitter object and trains it using provided data
    """
    skip = options.get('skip')
    model = Splitter(vectorizer.width, DIM, options=options)
    sys.stderr.write("Starting first epoch ...\n")
    for epoch in range(50):
        train_epoch(model, vectorizer, filename, epoch+1, skip=skip, options=options)   # This loads the best past model and trains for one epoch
        skip = None   # After the first epoch, don't skip batches


def test_splitter(vectorizer, testfile, options={}, verbose=False):
    """
    Instantiates a Splitter object, loads some weights, and tests it
    """
    model = Splitter(vectorizer.width, DIM, options=options)
    model.load_best()
    model.eval_mode()
    _monitor = gm.monitor_setup(testfile)   # For monitoring progress
    iterator = generate_batch_iterator(vectorizer, testfile, N=BATCH_SIZE, _monitor=_monitor)

    ALL = TP = TN = FP = FN = 0   # Counters for real stats
    # RTP = RTN = RFP = RFN = 0     # Counters for random comparison
    
    for batch in iterator:
        
        all, tp, tn, fp, fn, labels = model.predict_stats(batch)
        ALL += all
        TP  += tp
        TN  += tn
        FP  += fp
        FN  += fn

        """
        R  = model.predict_random(all)
        rtp, rtn, rfp, rfn = get_confusion(labels, R)
        RTP += rtp
        RTN += rtn
        RFP += rfp
        RFN += rfn
        """

    F1, Acc = compute_F1(TP, TN, FP, FN)
    print("\n\nAcc: %0.5f    F1: %0.5f    ALL: %d    TP: %d    TN: %d    FP: %d    FN: %d"% (Acc, F1, ALL, TP, TN, FP, FN))

    """
    if verbose:
        RF1, RAcc = compute_F1(RTP, RTN, RFP, RFN)
        print("\n\n(!RANDOM!)\nAcc: %0.5f    F1: %0.5f    TP: %d    TN: %d    FP: %d    FN: %d"% (RAcc, RF1, RTP, RTN, RFP, RFN))
    """


################################################################################
# MAIN

if __name__ == '__main__':
    parser = gm.argparser_ml({'desc': "Chinese Segmenter Model: model.py"})
    parser.add_argument('--get_class_weights', help='From the data get the average apriori weights of the classes', required=False, action='store_true')
    parser.add_argument('--fake', help='See if it can learn a simple function from fake data', required=False, action='store_true')
    args = parser.parse_args()   # Get inputs and options

    if args.train:
        vectorizer = gm.deserialize(VECTORIZER_FILE)
        train_splitter(vectorizer, TRAIN_FILE, options={'skip':args.skip, 'model_dir':args.model_dir, 'learning_rate':args.learning_rate, 'no_reload':args.no_reload, 'adaptive_learning_rate':args.adaptive_learning_rate, 'fake':args.fake, 'verbose':args.verbose})
        
    elif args.test:
        vectorizer = gm.deserialize(VECTORIZER_FILE)
        test_splitter(vectorizer, TEST_FILE, options={'model_dir':args.model_dir})

    elif args.get_class_weights:
        print("\n\nClass weights:", get_class_weights(TRAIN_FILE))
        
    else:
        print(__doc__)
        

################################################################################
################################################################################
