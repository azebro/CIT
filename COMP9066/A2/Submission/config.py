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
DATA_PATH = 'E:\\CHTBT\\data'#parent wehere the data files are stored. please keep all data files in single folder.
#CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'chat.txt'

DATA_PATH1 = 'E:\\CHTBT\\data\\cornell_movie_dialogs_corpus\\cornell movie-dialogs corpus'
CONVO_FILE1 = 'movie_conversations.txt'
LINE_FILE1 = 'movie_lines.txt'

JOEY = 'E:\\CHTBT\\data\\Friends_Analysis\\transcripts_friends\\season_all\\merged.csv'
OUTPUT_FILE = 'output_convo.txt'#file name of the stored conversations between user and bot
PROCESSED_PATH = 'processed'# parent path where thee preprocessed files will be stored.
CPT_PATH = 'E:\\CHTBT\\checkpoints' #model checkpoints path.

THRESHOLD = 2 #min number of occurence of a word to consider for adding it to vocab

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000 #test dataset size
#bucket size.
BUCKETS = [(16, 19),(40, 43)]#[(8, 10), (12, 14), (16, 19), (28, 28), (33, 33)]#, (40, 43), (50, 53), (60, 63)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3 #number of rnn cells
HIDDEN_SIZE = 512 # number of gru cells and embedding size.
BATCH_SIZE = 512# batch size.

LR = 0.001#learning rate
MAX_GRAD_NORM = 50.0

NUM_SAMPLES = 512 #number of sample to consider for sampled softmax.

ENC_VOCAB = 56135#encoder vocab size.
DEC_VOCAB = 52099#decoder vocab size.

