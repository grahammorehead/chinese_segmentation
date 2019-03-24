# chinese_segmentation
A Bi-directional LSTM to split Chinese text into words 

## INTRODUCTION
Chinese words comprise one or two characters (sometimes more). The fact that there are no spaces between words makes Chinese NLP more challenging than a language like English. Some research has shown that a classifier can be trained using merely a four-character window to decide if a given "interstice" (potential region between characters) should be a word boundary or not. These models have been known to reach 93% accuracy. The four-character window around a single interstice restricts that word-boundary information-- it can only propagate a distance of two characters.

## PURPOSE
This is not production code. The purpose of this code is to demonstrate an understanding of deep learning as it applies to natural language. In it, you will find a custom implementation of LSTMs, as opposed to using something off-the-shelf. Additionally, the code makes use of the module "gmutils," which I also wrote. From it I pulled in a custom loss function for skewed data.

## OBJECTIVE
To train an automatic word tokenizer for Chinese that goes beyond the 4-character window. A Stacked LSTM will be used to process an entire sentence at a time. The hidden states should carry information further than the two character limit mentioned above.

## THE DATA
- Each line of data comprises one sentence
- A double-space indicates a word boundary
- The training and test data are organized into separate files (under /data)
- It is assumed that each line is unrelated to other lines (I cannot read Chinese)

## CONSIDERATIONS
- We are constructing a "splitter," as opposed to the more complex task of Chinese word recognition. Functionally they are identical. This is the other side of the same coin.
- For characters in the test set that did not appear in the training set, the model must give a reasonable attempt
- This is not a meaning-representation problem but rather one of modeling the distributions of how characters interact to form words or not, i.e. word tokenization
- For memory reasons, only iterative approaches were considered (mini-batches of 1000), so it can run on a two-year old Macbook

## MODEL DESIGN
There are several components to the overall BiLSTM model, some of which were used only temporarily. 

### COMPONENTS
- Splitter (in model.py) The Python object that manages the other layers and objects
- BatchGraph (in model.py) The Python object that managed the tensor graph for a single batch of training data
- SimpleBiLSTM (in simple_lstm.py) The NN object that manages the LSTMs and the SplitLayer
- StackedBiLSTM (in stacked_lstm.py) The NN object that manages the stacked LSTMs and the SplitLayer (optional)
- LSTM (in stacked_lstm.py) My code for an LSTM cell
- SplitLayer (in split_layer.py) A NN object designed to incorporate all proved signals and
produce the final prediction
- SkewedL1Loss (in gmutils/gmutils/pytorch_utils.py) A custom loss function

### OPERATION
- Several threads run in parallel (limited by hardware)
- Each thread trains on distinct parts of the data:
  - loads a previous model from disk based on validation loss
  - trains on a batch
  - computes validation loss
  - saves model
- Separately, and periodically, the best model is tested on the held-out test set.

## USAGE

In order to try out this code, the following steps are suggested:

Assumes:
 - python > 3.6
 - Mac or Linux
```
python -m venv somenv
source somenv/bin/activate
git clone git@github.com:grahammorehead/gmutils.git
cd <THISDIR>
pip install -e ../gmutils
pip install -r requirements.txt
```

To construct a vectorizer:
`python vectorizer.py --train`


To train a model:
`python model.py --train --model_dir models --adaptive_learning_rate`


To test a model:
`python model.py --test --model_dir models`


Feel free to message me for any questions.
