import os
import re
import string
import camel_tools
from camel_tools.tokenizers.word import simple_word_tokenize

LANGUAGES_DICT = {'ALE':0,'ALG':1,'ALX':2,'AMM':3,'ASW':4,'BAG':5,
                'BAS':6,'BEI':7,'BEN':8,'CAI':9,'DAM':10,'DOH':11,
                  'FES':12,'JED':13,'JER':14,'KHA':15,'MOS':16,'MSA':17,
                  'MUS':18,'RAB':19,'RIY':20,'SAL':21,'SAN':22,'SFX':23,
                  'TRI':24,'TUN':25}
train_file = r"C:\CIT MSc Repo\CIT MSc in AI\COMP9066\A1\data\MADAR-Corpus-26-train.tsv"
dev_file = r"C:\CIT MSc Repo\CIT MSc in AI\COMP9066\A1\data\MADAR-Corpus-26-dev.tsv"

data_dir = r"\data"
#Define Emojis
RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
#Define Emoticons
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose 
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
#Remove noise chaaracters
regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]
#Arabic diacritics
arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

#Removing accented letters, diacritics and variations, 
# although I am not Arabic speaker...
def normalize_arabic(text):
    text = remove_diacritics(text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)

def remove_punctuations(text):
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؟'
    return ''.join(ch for ch in text if ch not in punctuations)

def remove_digits(text):
    remove_digits = str.maketrans('', '', string.digits)
    res = text.translate(remove_digits)
    return res

#Perform data cleaning based on the above
#Tokenisation is performed as well based on Camel Tools
def clean(corpus):
    new_corpus = []
    tokenised_corpus = []
    for sentence, dialect in corpus:
        clean_text = remove_punctuations(remove_punctuations(sentence))
        clean_text = normalize_arabic(clean_text)
        clean_text = remove_diacritics(clean_text)
        clean_text = remove_digits(clean_text)
        clean_text = strip_emoji(clean_text)
        tokenised_text = camel_tools.tokenizers.word.simple_word_tokenize(clean_text)
        new_corpus.append([clean_text, dialect])
        [tokenised_corpus.append([token, dialect]) for token in tokenised_text]
    return new_corpus, tokenised_corpus

#Read the MADAR file
def read_madar(file_name):
    sentence, label, rows= [],[], []
    with open(file_name, newline = '', encoding="utf8") as tsv: 
        print(file_name)
        madar_list=[x.strip().split('\t') for x in tsv]
    for row in madar_list:
        #save the array of sentence, dialect
        rows.append([row[0], row[1]]) 
        # save sentenceence
        sentence.append(row[0]) 
        # related dialect 
        label.append(row[1]) 
    return sentence, label, rows


sentences, dialects, rows = read_madar(dev_file)
print(set(dialects))

clean_rows = clean(rows)
