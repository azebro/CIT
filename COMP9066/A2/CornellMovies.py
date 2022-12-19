import numpy as np
import pandas as pd
import time
import os
import unicodedata

class CornellMovies:

    MOVIE_LINES_FIELDS = ["lineID","characterID","movieID","character","text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID","character2ID","movieID","utteranceIDs"]

    def __init__(self, folder):

        self.lines = {}
        self.conversations = []
        path = os.path.join(folder, "movie_lines.txt")
        print(path)

        self.lines = self.loadLines(os.path.join(folder, "movie_lines.txt"), self.MOVIE_LINES_FIELDS)
        self.conversations = self.loadConversations(os.path.join(folder, "movie_conversations.txt"), self.MOVIE_CONVERSATIONS_FIELDS)
        self.questions, self.answers = self.load_question_answers()


    def loadLines(self, fileName, fields):
        """
        Args:
            fileName (str): file to load
            field (set<str>): fields to extract
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        lines = {}

        with open(fileName, 'r', errors='ignore') as f:  
            for line in f:
                lines[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1].replace('\n', "")

        return lines
    
    def loadConversations(self, fileName, fields):
        '''
		
		Function made ONLY for Cornell dataset to extract conversations from the raw file.

	    '''
        conversations = []
        with open(fileName, 'r', errors='ignore') as f:
            for line in f.readlines():
                
                conversation = line.split(' +++$+++ ')[-1]
                conversation = conversation.replace("'", "")
                conversation = conversation[1:-2]
                conversation = conversation.split(", ")
                conversations.append(conversation)

        return conversations
        

    def load_question_answers(self):
        questions = []
        answers = []
        for i in range(len(self.conversations)):
            conversation = self.conversations[i]
            for line in range(len(conversation) - 1):
                questions.append(self.lines[conversation[line]])
                answers.append(self.lines[conversation[line + 1]])
        return questions, answers



    def getConversations(self):
        return self.conversations

    def get_question_answer(self):
        return self.questions, self.answers

