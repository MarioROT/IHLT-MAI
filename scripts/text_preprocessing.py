import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from typing import List, Dict, Tuple
import string

# Spacy imports 
import spacy
nlp = spacy.load('en_core_web_sm')

# TextServer imports
from complementary_material.textserver import TextServer

# NLTK imports
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
wnl = nltk.stem.WordNetLemmatizer()

class TextPreprocessing():

    def __init__(self,
                 data: List = False
                 ):
        self.data = data
        self.tokenized_data = []
        self.lemmatized_data = []
        self.lesk_lemmatized_data = []
        self.cleaned_data = []
        self.tag_conversor = {'NN': 'n', 'NNS': 'n', 'NNP':'n', 'NNPS':'n', 
                              'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 
                              'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ':'v',
                              'RB': 'r', 'RBR': 'r', 'RBS': 'r'}

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

    def lemmatize_data(self, data=False, method='nltk', verbose = True):
        ts = TextServer('user', 'passwd', 'tokenizer')
        self.lemmatized_data = []
        t_data = self.data if not data else data
        for sentence in t_data:
            if method == 'nltk':
                if not isinstance(sentence, list):
                    if verbose:
                        print('Applying NLTK tokenization to the sentence')
                    sentence = nltk.word_tokenize(sentence)
                tagged_pairs = nltk.pos_tag(sentence)
                lemmatization = lambda pair:wnl.lemmatize(pair[0], pos=self.tag_conversor[pair[1]]) if pair[1] in self.tag_conversor.keys() else pair[0]
                self.lemmatized_data.append([lemmatization(pair) for pair in tagged_pairs])
            elif method == 'spacy':
                doc = nlp(sentence if not isinstance(sentence, list) else ' '.join(sentence))
                self.lemmatized_data.append(token.lemma_ for token in doc)
            elif method == 'textserver':
                ts = TextServer('usuari', 'passwd', 'morpho')
                self.lemmatized_data([token[1] for token in ts.morpho(sentence)[0]])
        return self.lemmatized_data

    def lesk_lemmatize_data(self, data=False, method='nltk', verbose = True):
        wsd_sentences=[]
        self.lemmatized_data = []
        t_data = self.data if not data else data
        for sentence in t_data:
            if method == 'nltk':
                if not isinstance(sentence, list):
                    if verbose:
                        print('Applying NLTK tokenization to the sentence')
                    sentence = nltk.word_tokenize(sentence)
                self.lesk_lemmatized_data.append(self.lesk_lemmatize_sentence(sentence))
        return self.lesk_lemmatized_data

    def lesk_lemmatize_sentence(self, sentence):
        lemmatized_sentence =[]
        sentence_tagged = nltk.pos_tag(sentence)
        for (word,tag) in sentence_tagged:
            lemmatized_sentence.append(nltk.wsd.lesk(sentence, word, self.tag_conversor[tag] if tag in self.tag_conversor.keys() else None))
        return [syns if syns != None for syns in lemmatized_sentence]

    def clean_data(self, data = False, auto = True, lowercase = False, stopwords = False, minwords_len = False, signs = False):
        self.cleaned_data = []
        t_data = self.data if not data else data
        if auto:
            lowercase = True
            stopwords=set(nltk.corpus.stopwords.words('english'))
            signs = string.punctuation
            minwords_len = 2
            for element in t_data:
                self.cleaned_data.append(self.clean_sentence(element, lowercase, stopwords, minwords_len, signs))
        else: 
            for element in t_data:
                 self.cleaned_data.append(self.clean_sentence(element, lowercase, stopwords, minwords_len, signs))
        return self.cleaned_data

    def clean_sentence(self,sentence, lowercase = True, stopwords = False, minwords_len = False, signs = False):
        sentence = sentence.split(' ')
        if lowercase:
            sentence = [word.lower() for word in sentence]
        if signs:
            sentence = [word if not any(caracter in signs for caracter in word) else self.remove_signs(word, signs) for word in sentence]
        if stopwords:
            sentence = [word for word in sentence if word not in stopwords and word.isalpha()]
        if minwords_len:
            sentence = [word for word in sentence if len(word) > minwords_len]
        return sentence

    @staticmethod
    def remove_signs(wrd,signs):
        wrd = list(wrd)
        wrd = [word for word in wrd if not any(caracter in signs for caracter in word)]
        wrd = ''.join(wrd)
        return wrd
