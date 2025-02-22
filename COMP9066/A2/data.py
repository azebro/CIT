""" A neural chatbot using sequence to sequence model with
attentional decoder. 
This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
This file contains the code to do the pre-processing for the
Cornell Movie-Dialogs Corpus.
See readme.md for instruction on how to run the starter code.
"""
import os
import random
import re
import demoji
import numpy as np

import unicodedata 
import config
import pandas as pd
import json
from collections import Counter

import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
import preprocessor as p
from ekphrasis.classes.segmenter import Segmenter
seg_tw = Segmenter(corpus="twitter")


def preprocess_data(data):
    #Removes Numbers
    
    data = data.astype(str).str.replace('\d+', '')
    lower_text = data.str.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer =  TweetTokenizer()

    def lemmatize_text(text):
        return [(lemmatizer.lemmatize(w)) for w \
                        in w_tokenizer.tokenize((text))] 

    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', (word))
            if new_word != '':
                new_words.append(new_word)
        return new_words 
    words = lower_text.apply(lemmatize_text)
    words = words.apply(remove_punctuation)
    stop_words = set(stopwords.words('english'))
    remove_stop = lambda x: [item for item in x if item not in stop_words]
    words_no_stop = remove_stop(words)

    return pd.DataFrame(words_no_stop)

def get_twitter_lines():
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    print(config.LINE_FILE)
    with open(file_path, 'r', errors='ignore') as f:
        lines = f.readlines()
        df = pd.DataFrame(lines, columns=["text"])
        words = preprocess_data(df["text"])
        df["text"] = words
        stop_words = set(stopwords.words('english'))
        df["text"] = df["text"].apply(lambda x: [item for item in x if item not in stop_words])
        


 


def get_lines():
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    print(config.LINE_FILE)
    with open(file_path, 'r', errors='ignore') as f:
        '''
        lines = f.readlines()
        df = pd.DataFrame(lines, columns=["text"])
        words = preprocess_data(df["text"])
        '''
        # for line in lines:
        i = 0
        ques = []
        ans = []
        try:
            for line in f:
                #line = demoji.replace(string= line, repl= "")
                #parts = line.split('\n')
                
                line = normalizeString(line)
                if i % 2 == 0:
                    
                    ques.append(line.rstrip("\n"))
                else:
                    ans.append(line.rstrip("\n"))
                i += 1
        except UnicodeDecodeError:
            print(i, line)
    ques1 = demoji.replace(string= ques[11], repl= "")
    print(ques1)
    print(ques[10:20])
    print(ans[10:25])
    return ques,ans


def get_lines2():
    id2line = {}
    file_path = os.path.join(config.DATA_PATH1, config.LINE_FILE1)
    print(config.LINE_FILE1)
    with open(file_path, 'r', errors='ignore') as f:
        # lines = f.readlines()
        # for line in lines:
        i = 0
        try:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) == 5:
                    if parts[4][-1] == '\n':
                        parts[4] = parts[4][:-1]
                    id2line[parts[0]] = parts[4]
                i += 1
        except UnicodeDecodeError:
            print(i, line)
    return id2line


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'\.+', r".", s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def joey_data():
    lst1 = []
    file_path = os.path.join(config.DATA_PATH, config.JOEY)

    with open(file_path, "r+",encoding="utf-8", errors='ignore') as fp:
        for cnt, line in enumerate(fp):
            if line.startswith('[') or line.startswith('\n') or line.startswith('(') :
                    continue
            else:
                
                lst1.append(line)
                    
    ques = []
    ans = []
    for i in range(len(lst1)):
        lst1[i] = re.sub(r"\(.*\)","", lst1[i])
        lst1[i] = re.sub(r'\.+', "", lst1[i])
        if 'Joey:' in lst1[i]:
            
          if len(lst1[i-1].split(':'))== 1:
            ques.append(lst1[i-1].rstrip("\n"))
            
          else:
              ques.append(lst1[i-1].split(':')[1].rstrip("\n"))
          
          if len(lst1[i].split(':')[1])==1:
              ans.append(lst1[i].rstrip("\n"))
          else:
              ans.append(lst1[i].split(':')[1].rstrip("\n"))
    print('len of dialouges',len(ans))
    return ques,ans

def get_convos():
    """ Get conversations from the raw data """
    file_path = os.path.join(config.DATA_PATH1, config.CONVO_FILE1)
    convos = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos

def question_answers(id2line, convos):
    """ Divide the dataset into two sets: questions and answers. """
    questions, answers = [], []
    for convo in convos:
        for index, line in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]])
            answers.append(id2line[convo[index + 1]])
    assert len(questions) == len(answers)
    return questions, answers

def prepare_dataset(questions, answers):
    # create path to store all the train & test encoder & decoder
    make_dir(config.PROCESSED_PATH)
    
    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(questions))],config.TESTSET_SIZE)
    
    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED_PATH, filename),'w',encoding='utf-8', errors='ignore'))

    for i in range(len(questions)):
        if i in test_ids:
            files[2].write(questions[i]+ '\n')
            files[3].write(answers[i]+ '\n')
        else:
            files[0].write(questions[i]+ '\n')
            files[1].write(answers[i]+ '\n')

    for file in files:
        file.close()

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    line = normalizeString(line)
    #line = demoji.replace(string= line, repl= "")
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):

            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words

def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    vocab = {}
    with open(in_path, 'r',encoding = 'utf-8') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w',encoding = 'utf-8') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n') 
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                break
            f.write(word + '\n')
            index += 1
        with open('config.py', 'a') as cf:
            if filename[-3:] == 'enc':
                cf.write('ENC_VOCAB = ' + str(index) + '\n')
            else:
                cf.write('DEC_VOCAB = ' + str(index) + '\n')

def load_vocab(vocab_path):
    with open(vocab_path, 'r',encoding = 'utf-8') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]

def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'r',encoding='utf-8')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w',encoding='utf-8')
    
    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec': # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        if mode == 'dec':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')
    questions, answers = get_lines()
    questions1, answers1 = joey_data()
    id2line = get_lines2()

    convos1 = get_convos()
    questions2, answers2 = question_answers(id2line, convos1)
    #prepare_dataset(questions+questions1, answers+answers1)
    #print(len(questions),len(questions1),len(questions2))
    #print(len(answers),len(answers1),len(answers2))
    ques = questions2+questions1+questions
    ans = answers2+answers1+answers

    for i in range(len(ques)):
      ques[i] = normalizeString(ques[i])
      ans[i] = normalizeString(ans[i])
    #print(len(ques),len(ans))
    prepare_dataset(ques, ans)

def process_data():
    print('Preparing data to be model-ready ...')
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'r')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'r')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets

def _pad_input(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

if __name__ == '__main__':
    prepare_raw_data()
    process_data()