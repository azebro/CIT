""" A neural chatbot using sequence to sequence model with
attentional decoder. 
This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
This file contains the hyperparameters for the model.
See README.md for instruction on how to run the starter code.
"""

# parameters for processing the dataset
DATA_PATH = 'E:\\CIT MSc Repo\\CIT MSc in AI\\COMP9066\\A2\\data'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'cleaned_corpus_en.txt'

DATA_PATH1 = 'E:\\CIT MSc Repo\\CIT MSc in AI\\COMP9066\\A2\\data\\cornell movie-dialogs corpus'
CONVO_FILE1 = 'movie_conversations.txt'
LINE_FILE1 = 'movie_lines.txt'

JOEY = 'friends.csv'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'E:\\nlp\\checkpoints'

THRESHOLD = 20

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000
#[(8, 10), (12, 14), (16, 19)]
BUCKETS = [(8, 10), (12, 14), (16, 19), (28, 28), (33, 33)]#, (40, 43), (50, 53), (60, 63)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
'''
ENC_VOCAB = 69652
DEC_VOCAB = 65954
ENC_VOCAB = 69182
DEC_VOCAB = 66155
ENC_VOCAB = 56202
DEC_VOCAB = 52125
ENC_VOCAB = 57725
DEC_VOCAB = 52668


ENC_VOCAB = 56122
DEC_VOCAB = 52117
'''

ENC_VOCAB = 132143
DEC_VOCAB = 115928
