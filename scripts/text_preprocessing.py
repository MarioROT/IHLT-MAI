import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from typing import List, Dict, Tuple

# Spacy imports 
import spacy
nlp = spacy.load('en_core_web_sm')

# TextServer imports
from complementary_material.textserver import TextServer

# NLTK imports
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessing():

    def __init__(self,
                 data: List = False
                 ):
        self.data = data
        self.tokenized_data = []
        self.lemmatized_data = []
        self.tag_conversor = {'NN': 'n', 'NNS': 'n', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 
                              'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 
                              'VBZ': 'v', 'RB': 'r', 'RBR': 'r', 'RBS': 'r'}

    def __len__(self):
        return len(self.data)

    def tokenize_data(self, data=False, method='nltk'):
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

    def lemmatize_data(self, data=False, method='nltk'):
        ts = TextServer('user', 'passwd', 'tokenizer')
        self.lemmatized_data = []
        t_data = self.data if not data else data
        for sentence in t_data:
            if method == 'nltk':
                if not isinstance(sentence, list):
                    print('Applying NLTK tokenization to the sentence')
                    sentence = nltk.word_tokenize(sentence)
                tagged_pairs = nltk.pos_tag(sentence)
                lemmatization = lambda pair:wnl.lemmatize(pair[0], pos=self.tag_conversor[pair[1]]) if pair[1] in self.tag_conversor else pair[0]
                self.lemmatized_data.append([lemmatization(pair) for pair in tagged_pairs])
            elif method == 'spacy':
                doc = nlp(sentence)
                self.lemmatized_data.append([token.lemma_ for token in doc])
            elif method == 'textserver':
                ts = TextServer('usuari', 'passwd', 'morpho')
                self.lemmatized_data([token[1] for token in ts.morpho(sentence)[0]])
        return self.lemmatized_data




