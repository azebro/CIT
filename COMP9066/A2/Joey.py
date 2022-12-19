import numpy as np
import pandas as pd
import time
import os
import unicodedata
from convokit import Corpus, download


class Joey:

    def __init__(self):
        self.questions, self.answers = self.load_question_answers()


    def load_question_answers(self):

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

    def get_question_answer(self):
        return self.questions, self.answers