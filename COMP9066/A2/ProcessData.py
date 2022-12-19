# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:33:10 2020

@author: adam
"""

import numpy as np

import CornellMovies
import Joey
import Twitter
import config


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
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
import preprocessor as p
from convokit import Corpus, download


EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\''

def filter_line(line, blacklist):
    return ''.join([ ch for ch in line if ch not in blacklist])


def extractText(self, line, isTarget=False):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
            isTarget (bool): Define the question on the answer
        Return:
            list<int>: the list of the word ids of the sentence
        """
        words = []

        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if not isTarget:
                i = len(sentencesToken)-1 - i

            tokens = nltk.word_tokenize(sentencesToken[i])

            # If the total length is not too big, we still can add one more sentence
            if len(words) + len(tokens) <= self.args.maxLength:
                tempWords = []
                for token in tokens:
                    tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

                if isTarget:
                    words = words + tempWords
                else:
                    words = tempWords + words
            else:
                break  # We reach the max length already

        return words


def preprocess_data(data, isTarget = False):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer =  TweetTokenizer(strip_handles=True, reduce_len=True)
    stop_words = set(stopwords.words('english'))
    clean_words = []
    

    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', (word))
            if new_word != '':
                new_words.append(new_word)
        return new_words 

    lines = []
    for line in data:
        words = []
        sentencesToken = nltk.sent_tokenize(line)
        
        

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if not isTarget:
                i = len(sentencesToken)-1 - i

            sent = sentencesToken[i].lower()
            sent = filter_line(sent, EN_BLACKLIST)
            tokens = [(lemmatizer.lemmatize(w)) for w in nltk.word_tokenize(sent)]
            tokens = [item for item in tokens if item not in stop_words]
            tokens = remove_punctuation(tokens)
            if len(words) + len(tokens) <= 10:
                temp_words = []
                for token in tokens:
                    temp_words.append(token)
                if isTarget:
                    words = words + temp_words
                else:
                    words = temp_words + words
            else:
                break
        lines.append(words)
    return lines

def filter_empty(questions, answers):
    question_tokens = []
    answer_tokens = []
    for i in range (len(questions)):
        if len(questions[i]) > 0 and len(answers[i]) > 0:
            question_tokens.append(questions[i])
            answer_tokens.append(answers[i])
    assert len(question_tokens) == len(answer_tokens)
    return question_tokens, answer_tokens


def create_vocabulaty(questions, answers):
    assert len(questions) == len(answers)
	vocab = []
    for i in range(len(questions)):
		
		for word in questions[i]:
			vocab.append(word)
		
		for word in answers[i]:
			vocab.append(word)
    vocab = Counter(vocab)
    new_vocab = []
	for key in vocab.keys():
		if vocab[key] >= config.VOCAB_THRESHOLD:
			new_vocab.append(key)

	new_vocab = ['<PAD>', '<GO>', '<UNK>', '<EOS>'] + new_vocab
    word_to_id = {word:i for i, word in enumerate(new_vocab)}
	id_to_word = {i:word for i, word in enumerate(new_vocab)}

	return new_vocab, word_to_id, id_to_word

def pad_data(data, word_to_id, max_len, target=False):
	'''
		
		If the sentence is shorter then wanted length, pad it to that length

	'''
	if target:
		return data + [word_to_id['<PAD>']] * (max_len - len(data))
	else:
		return [word_to_id['<PAD>']] * (max_len - len(data)) + data
        

def bucket_data(questions, answers, word_to_id):

	'''
	
		If you prefere bucketing version of the padding, use this function to create buckets of your data.

	'''
	assert len(questions) == len(answers)

	bucketed_data = []
	already_added = []
	for bucket in config.BUCKETS:
		data_for_bucket = []
		encoder_max = bucket[0]
		decoder_max = bucket[1]
		for i in range(len(questions)):
			if len(questions[i]) <= encoder_max and len(answers[i]) <= decoder_max:
				if i not in already_added:
					data_for_bucket.append((pad_data(questions[i], word_to_id, encoder_max), pad_data(answers[i], word_to_id, decoder_max, True)))
					already_added.append(i)

		bucketed_data.append(data_for_bucket)

	return bucketed_data
        


jo = Joey.Joey()
jq, ja = jo.get_question_answer()
jtq = preprocess_data(jq)
jta = preprocess_data(ja)




cm = CornellMovies.CornellMovies(".//data//cornell movie-dialogs corpus/")
q, a = cm.load_question_answers()
aa = preprocess_data(q)
bb = preprocess_data(q, True)

