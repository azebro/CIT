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
import demoji
demoji.download_codes()
import unicodedata 
import config
import pickle

#config.datapath is where all the data files should be stores

def get_lines():
	#reading function for TWITTER chat corpus.
	#config.LINE_FILE will be the name of the twitter data file named 'chat.txt'.
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    print('loading twitter corpus',config.LINE_FILE)
    with open(file_path, 'r', errors='ignore') as f:
  		#iteratig through corpus
        i = 0
        ques = []
        ans = []
        try:
            for line in f:
                #taking each line in the corpus.
                #pre-processing the line using normalizeString function.
                line = normalizeString(line)
                if i % 2 == 0:
                	#dividing corpus on line number if the line is even then its in question category.
                    
                    ques.append(line.rstrip("\n"))
                else:
                	#if line number is odd then it is the reply to the question.
                    ans.append(line.rstrip("\n"))
                i += 1
        except UnicodeDecodeError:
        	#
            print(i, line)
    #demoji will remove all the emoji's in the text if any. 

    return ques,ans


def get_lines2():
	# reading cornell_movie_dialogs_corpus corpus.
    id2line = {}
    file_path = os.path.join(config.DATA_PATH1, config.LINE_FILE1)
    print('loading cornell_movie_dialogs_corpus',config.LINE_FILE1)
    with open(file_path, 'r', errors='ignore') as f:
        
        i = 0
        try:
            for line in f:
                parts = line.split(' +++$+++ ')#splitting lines by the given string to get questions and answers.#you can have a look at the corpus file for better understanding.
                if len(parts) == 5:
                    if parts[4][-1] == '\n':
                        parts[4] = parts[4][:-1]
                    id2line[parts[0]] = parts[4]
                i += 1
        except UnicodeDecodeError:
            print(i, line)
    return id2line

def unicodeToAscii(s):
	#converting unicode to ascii string.
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'\.+', r".", s)#removing extra full stops.
    s = re.sub(r"([.!?])", r" \1", s) #inserting spacing when ever ".","!","?", appear.
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s) #replacing string which is not numbers or numbers except ".","!","?"
    s = re.sub(r"\s+", r" ", s).strip()#string extra spaces.
    return s

def joey_data():
	#reading friends data
    lst1 = []
    file_path = os.path.join(config.DATA_PATH, config.JOEY)
    print('reading friends corpus',file_path)
    with open(file_path, "r+",encoding="utf-8", errors='ignore') as fp:
        for cnt, line in enumerate(fp):
        	#if line starts with "[",'\n','(' it will skip that line, because it contains explaination rather than dialouge.
            if line.startswith('[') or line.startswith('\n') or line.startswith('(') :
                    continue
            else:
                
                lst1.append(line)
                    
    ques = []
    ans = []
    for i in range(len(lst1)):
        lst1[i] = re.sub(r"\(.*\)","", lst1[i])#removing  data within '()' 
        lst1[i] = re.sub(r'\.+', "", lst1[i]) #removing extra full stops.
        if 'Joey:' in lst1[i]:
          #getting line which start with 'joey' as the answer.
          #the line before that will be the question.
          #we are taking just joey data because we want to make our chatbot have personality.
          #at the end we will only we will just train the model on this dataset only for 3000-4000 iterations.

            
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

def prepare_raw_data(feedback = False):
	#THIS IS THE MAIN FUNCTION WHERE WE CAN CHOOSE WHICH DATASET TO CHOOSE OR NOT.
	#IN THE BEGINNING WE WILL CHOOSE WHOLE DATA 
	#BUT AT THE END WE WILL ONLY USE THE DATA FROM JOEY_DATA() FUNCTIOIN AND PASS IT TO prepare_dataset() .
	# if you want to just train the data on just joey_data , uncomment the below and comment the exiting one.

    '''
    
    questions1, answers1 = joey_data()
    if feedback ==True :
    dbfile = open('feedback', 'rb')
    db = pickle.load(dbfile) 
        #print(db)
    questions4 = []
    answers4 = []
    for i in db:
        questions4.append(i[0])
        answers4.append(i[1])
    ques = questions4+questions2+questions1+questions
    ans = answers4+answers2+answers1+answers
    else:
        ques = questions2+questions1+questions
        ans = answers2+answers1+answers
    
    for i in range(len(ques)):
    ques[i] = normalizeString(ques[i])
    ans[i] = normalizeString(ans[i])
    #print(len(ques),len(ans))
    prepare_dataset(ques, ans)
    '''
    

    print("Preparing raw data into train set and test set ...")

    questions, answers = get_lines()
    questions1, answers1 = joey_data()
    id2line = get_lines2()

    convos1 = get_convos()
    questions2, answers2 = question_answers(id2line, convos1)
    

    if feedback ==True :
      dbfile = open('feedback', 'rb')
      db = pickle.load(dbfile) 
		  #print(db)
      questions4 = []
      answers4 = []
      for i in db:
          questions4.append(i[0])
          answers4.append(i[1])
      ques = questions4+questions2+questions1+questions
      ans = answers4+answers2+answers1+answers
    else:
    	ques = questions2+questions1+questions
    	ans = answers2+answers1+answers
    
    for i in range(len(ques)):
      ques[i] = normalizeString(ques[i])
      ans[i] = normalizeString(ans[i])
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