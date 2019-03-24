""" utils.py

    Helper functions

"""
import sys
import os, re
import time
from gmutils import err
import gmutils as gm
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import pandas as pd

################################################################################
# SETTINGS

ENCODING = 'big5hkscs'
DATADIR  = 'cache'

################################################################################
# EXCEPTIONS

class NaN_Exeption(Exception):
   """ The tensor has NaNs! """
   pass

################################################################################
# FUNCTIONS

def generate_file_iterator(filename, _monitor=None):
    """
    Creates an easy to use file iterator, yielding a list of strings representing each line.  Where each string is a word.
    """
    line = []              # a gathered list of words on one line, where each "word" is one or two characters
    word = None            # a string representing a whole word, often comprising multiple characters
    seen = set([])

    with open(filename, "rb") as f:
        for nextline in f:
            if _monitor:
                gm.monitor(_monitor)    # print progress to the command line
                
            try:
                characters = nextline.decode(ENCODING, "strict")    # The characters in one line
            except:
                continue

            for c in characters: ### Iterate over characters ###
                if ord(c) == 12288:   # Normalize ideographic spaces
                    c = ' '

                if c == '\n':   ## This line is done
                    if word is not None:
                        line.append(word)
                    yield line
                    line = []
                    word = None
                    
                else:           ## Gather the next character into a word
                    if c == ' '  and  last_c == ' ':   # An in-line word separator -- This word is done
                        if word is not None:
                            line.append(word)
                            word = None
                            
                    elif c == ' ':                     # A character separator
                        pass
                    
                    elif word is None:                 # Just starting to gather the next word
                        word  = c
                            
                    else:
                        word += c                      # Adding a new character to the current word

                last_c = c      ## Remember the previous character

        if word is not None:
            line.append(word)

        if len(line):
            yield line


def generate_batch_iterator(vectorizer, filename, N=1000, _monitor=None):
    """
    Generates training batches of N samples each
    """
    output = []
    iterator = generate_file_iterator(filename, _monitor=_monitor)
    for sample in iterator:
        if _monitor.get('ignore'):
            yield None
            continue

        """ Here we gather all labels into a single one-dimensional array.  The only thing we know about its length is that it will be identical
            to the length of the prediction array.  The labels will be an array of 1s (separation) and 0s (no separation).
        """
        labels  = get_labels_from_sample(sample)        # Gather all labels into a single array
        vectors = vectorizer.vectorize_sample(sample)   # Vectorize (discarding all labels)
        item = (labels, vectors)                        # Each element in the batch list will be this tuple
        output.append(item)
        
        if len(output) >= N:
            yield output
            output = []
            
    if len(output) > 0:
        yield output   # don't forget stragglers


def generate_blind_iterator(filename):
    """
    Generate an iterator for a file which deletes all training labels (the splits) while passing through the actual characters.
    """
    iterator = generate_file_iterator(filename)
    for line in iterator:
        yield ' '.join(line)


def get_vectors_for_collation(i, vectors):
    """
    Collate vectors to be used in rows of the output dataframe (see 'collate_for_4gram_model')
    """
    v1 = vectors[i]
    v2 = vectors[i+1]
    v3 = vectors[i+2]
    v4 = vectors[i+3]

    return v1, v2, v3, v4
    
        
def collate_for_4gram_model(labels, vectors):
    """
    (this was a separate line of research not used by the current model)

    The input data format for this model requires a Series and a Dataframe lined up with each other, where each element in the Series
    corresponds to one row in the Dataframe.  That one row will represent the vectors that come before and after the potential split.

    This differs from the incoming data format in that the incoming data is of variable length, representing rows of data from an inhomogenous
    dataset.

    INCOMING:
        (The following example would have been generated from four characters, some grouped, e.g.: ['A', 'BC', 'D']
        labels  = [1, 0, 1]             (The 0 in the second element corresponds to the fact that 'BC' is not split)
        vectors = [vA, vB, vC, vD]      (Each original character is represented by a vector)

    OUTPUT:
        v0      = vector of all zeros   (represents null area before or after the characters)
        labels  = [1, 0, 1]
        vectors = [(v0+vA, vB+VC), (vA+vB, vC+vD), (vB+vC, vD+v0)]

        The 'vectors' list above comprises three final vectors, each double the width of the original vector.
        In the function, these will be called: v1, v2, v3, v4

    Parameters
    ----------
    labels : list of int (either 0 or 1)
    
    vectors : list of vectors in csr format (sparse)

    v0      : sparse vector full of zeros
    """
    output = None
    for i, label in enumerate(labels):   # create one output vector for each label
        v1, v2, v3, v4 = get_vectors_for_collation(i, vectors)
        vec = hstack([v1+v2, v3+v4])
        if output is None:
            output = vec
        else:
            output = vstack([output, vec])

    return output
    

def generate_sklearn_batch_iterator(vectorizer, filename, N=1000, _monitor=None, options={}):
    """
    (separate line of research not used for the current model)

    Generates training batches of N samples each, but collates them in the manner expected by Sci-kit Learn
    """
    batch_num    = 1
    batch_file   = DATADIR + '/batch_%d.gz'% batch_num
    out_labels   = None                                  # To be yielded when ready
    out_vectors  = None
    v0           = csr_matrix( [0]*vectorizer.width )    # Create a sparse v0 to be used in collation (see 'collate_for_4gram_model' above)
    iterator     = generate_file_iterator(filename, _monitor=_monitor)
    total        = 0
    skip_i       = options.get('skip_i')                 # Skip ahead to this line
    
    for sample in iterator:
        """ Here we gather all labels and vectors into the format needed by the 4-gram model
        """
        total      += 1
        try:
            if _monitor['i'] < skip_i:    # To quickly get back to where you were
                continue
        except:
            pass
        if options.get('max_lines'):
            if total > options.get('max_lines'):
                continue
            
        labels      = get_labels_from_sample(sample)             # Gather all labels into a single array
        vectors     = vectorizer.vectorize_sample(sample)        # Vectorize (discarding all labels)
        vectors     = [v0] + vectors + [v0]                      # Pad with zero vectors
        vectors     = collate_for_4gram_model(labels, vectors)   # Convert to collated format needed by 4-gram model
        labels      = pd.Series(labels)                          # Pandas format  (Tried sparse, but not compatible with sklearn)
        vectors     = vectors.todense()
        vectors     = pd.DataFrame(vectors)                      # Pandas format
        #vectors    = pd.SparseDataFrame(vectors)                # Sparse Pandas format
        #vectors.fillna(0, inplace=True)

        if out_labels is None:
            out_labels  = labels
            out_vectors = vectors
        else:
            out_labels  = pd.concat([out_labels, labels], ignore_index=True)     # Gather with other samples to accrue a sizable batch
            out_vectors = pd.concat([out_vectors, vectors], ignore_index=True)
        assert(out_labels.shape[0] == out_vectors.shape[0])                      # Confirm same number of labels / vectors
        
        if out_labels is not None  and  len(out_labels) >= N:
            batch = (out_labels, out_vectors)
            yield batch
            # gm.json_dump_gz(batch_file, batch)   # Save batch to cache for faster retrieval
            
            # Refresh vars
            batch_num += 1
            batch_file = DATADIR + '/batch_%d.gz'% batch_num
            out_labels  = None
            out_vectors = None
            
    if out_labels is not None  and  len(out_labels) > 0:
        batch = (out_labels, out_vectors)
        yield batch
    

def get_confusion(labels, preds):
    """
    Get the binary confusion matrix values for two equal-length lists
    """
    try:
        assert(len(labels) == len(preds))
    except:
        err([labels, preds, len(labels), len(preds)])
        
    TP = TN = FP = FN = 0
    
    ##  y=label, x=pred  ##
    for i, y in enumerate(labels):
        x = preds[i]
        
        if y == 1  and  x == 1:
            TP += 1
        elif y == 0  and  x == 0:
            TN += 1
        elif y == 0  and  x == 1:
            FP += 1
        elif y == 1  and  x == 0:
            FN += 1

    return TP, TN, FP, FN
    

def get_labels_from_sample(sample):
    """
    Each label of Chinese words having at most N-1 elements, assuming that it contains N characters that may be grouped.

    Parameters
    ----------
    sample : list of N characters

    Returns
    -------
    list of N-1 float on [0,1] (0 represents no split)
    """
    labels = []
    for word in sample:
        if len(word) > 1:
            for _ in range(len(word)-1):
                labels.append(0)    # within a word, append a '0' for each interstice
            labels.append(1)   # at the end of a word, append a '1'
        else:
            labels.append(1)
            
    labels = labels[:-1]   # Throw away the last value, it doesn't represent an interstice
    
    return labels


def test_label_getter(sample):
    """
    Quick sanity check for 'get_labels_from_sample()'
    """
    print()
    for word in sample:
        print("\t", word)
    labels = get_labels_from_sample(sample)
    print(labels)
    print("\nLength of sample:", len(sample))
    print("Number of chars: ", len(''.join(sample)))
    print("Number of labels: ", len(labels))


def get_class_weights(filename):
    """
    Get the relative occurrence weight for each class in the data
    """
    _monitor = gm.monitor_setup(filename)   # For monitoring progress
    iterator = generate_file_iterator(filename, _monitor=_monitor)
    N = 0
    totals = [0, 0]
    for sample in iterator:
        labels  = get_labels_from_sample(sample)   # Gather all labels into a single array
        n = len(labels)
        t = sum(labels)
        f = n - t
        N += n
        totals[0] += f
        totals[1] += t

    return [totals[0]/N, totals[1]/N]
    

def compute_F1(TP, TN, FP, FN):
    """
    Return the F1 score
    """
    numer = 2 * TP
    denom = 2 * TP + FN + FP
    F1 = numer/denom
    Acc = 100. * (TP + TN) / (TP + TN + FP + FN)
    
    return F1, Acc


def zeros_only(preds, labels):
    """
    Return only those samples having a label=0
    """
    out_preds  = []
    out_labels = []
    for i, label in enumerate(labels):
        if label == 0:
            out_labels.append(label)
            out_preds.append( preds[i] )
            
    return out_preds, out_labels
        

def vec_to_str(vec):
    """
    Convert a vec (of supposedly integers) into a str
    """
    return ''.join(map(str, map(int, vec)))


################################################################################
# MAIN

if __name__ == '__main__':
    parser = gm.argparser_ml({'desc': "Segmenter Utilities: util.py"})
    parser.add_argument('--test_label', help='Test sample->labels function', required=False, action='store_true')
    args = parser.parse_args()   # Get inputs and options

    if args.test_label:
        sample = ['大多數', '研究', '廣義', '相對論', '的', '物理學家', '不', '相信', '反張量', '能', '給出', '一', '個', '好', '的', '＊', '局部', '的', '（', 'ｌｏｃａｌ', '）', '＊', '能量', '密度', '的', '定義', '，']
        test_label_getter(sample)
        sample = ['與', '空氣', '中', '相反', '．']
        test_label_getter(sample)
        
    else:
        print(__doc__)
        

################################################################################        
################################################################################

