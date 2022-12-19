import numpy as np
import pandas as pd
import time
import os
import unicodedata


class Twitter:

    def __init__(self, folder, file):

        self.questions, self.answers = self.load_question_answers(os.path.join(folder, file))


    def load_question_answers(self, path):
        with open(path, 'r', errors='ignore') as f:
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

    def get_question_answer(self):
        return self.questions, self.answers