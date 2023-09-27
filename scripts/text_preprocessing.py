import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from typing import List, Dict, Tuple

import spacy
nlp = spacy.load('en_core_web_sm')

from complementary_material.textserver import TextServer

import nltk
nltk.download('punkt')

class TextPreprocessing():

    def __init__(self,
                 data: List = False
                 ):
        self.data = data
        self.tokenized_data = []

    def __len__(self):
        return len(self.data)

    def tokenize_data(self, data = False, method='nltk'):
        ts = TextServer('user', 'passwd', 'tokenizer') 
        self.tokenized_data = []
        t_data = self.data if not data else data
        for sentence in t_data:
            if method == 'nltk':
                self.tokenized_data.append(nltk.word_tokenize(sentence))
            elif method == 'spacy':
                doc = nlp(sentence)
                self.tokenized_data.append([[token.text for token in sent] for sent in doc.sents][0])
            elif method == 'textserver':
                self.tokenized_data.append(ts.tokenizer(sentence))
        return self.tokenized_data


