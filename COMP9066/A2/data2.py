
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
from convokit import Corpus, download



def preprocess_data(data):
    #Removes Numbers
    
    data = data.astype(str).str.replace('\d+', '')
    lower_text = data.str.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer =  TweetTokenizer(strip_handles=True, reduce_len=True)

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
    words = lower_text.apply(lambda x: p.clean(x))
    words = words.apply(lemmatize_text)
    words = words.apply(remove_punctuation)
    stop_words = set(stopwords.words('english'))
    remove_stop = lambda x: [item for item in x if item not in stop_words]
    words_no_stop = remove_stop(words)

    return pd.DataFrame(words_no_stop)


def get_qa_lines():

    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    print(config.LINE_FILE)
    with open(file_path, 'r', errors='ignore') as f:
         # for line in lines:
        i = 0
        ques = []
        ans = []
        try:
            for line in f:
                if i % 2 == 0:
                    ques.append(line.rstrip("\n"))
                else:
                    ans.append(line.rstrip("\n"))
                i += 1
        except UnicodeDecodeError:
            print(i, line)
    return ques,ans

def get_film_lines():
    id2line = {}
    file_path = os.path.join(config.DATA_PATH1, config.LINE_FILE1)
    print(config.LINE_FILE1)
    with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
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


def get_joey_data():
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

def get_joey_lines():
    corpus = Corpus(filename=download("friends-corpus"))
    list_of_friends = list(corpus.utterances.values())
    fl = [u for u in list_of_friends if u.speaker.id == 'Joey Tribbiani']
    qs = []
    an = []
    for line in fl:
        joey_text = line.text
        question = line.reply_to
        if question is not None:
            q = corpus.utterances[question].text
            qs.append(q)
            an.append(joey_text)
    return qs, an


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



def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')
    print("Getting Twitter lines")
    questions, answers = get_qa_lines()
    print("Getting Joey lines")
    questions1, answers1 = get_joey_data()
    print("Getting film lines")
    id2line = get_film_lines()
    convos1 = get_convos()
    questions2, answers2 = question_answers(id2line, convos1)
    print("All lines loaded")
    #prepare_dataset(questions+questions1, answers+answers1)
    #print(len(questions),len(questions1),len(questions2))
    #print(len(answers),len(answers1),len(answers2))
    q, a = get_joey_lines()
    ques = questions2+questions1+questions
    ans = answers2+answers1+answers
    #d = {"Question": ques, "Answer": ans}
    d = {"Question": q, "Answer": a}
    df = pd.DataFrame(d)
    print("Pre-process questions")
    processed_questions = preprocess_data(df["Question"])
    print("Pre-process answers")
    processed_answers = preprocess_data(df["Answer"])



    
    #print(len(ques),len(ans))
    prepare_dataset(ques, ans)


if __name__ == '__main__':
    
    prepare_raw_data()
    #process_data()