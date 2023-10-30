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
nltk.download('treebank')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('wordnet_ic')
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
wnl = nltk.stem.WordNetLemmatizer()
from nltk.tree import Tree

class TextPreprocessing():

    def __init__(self,
                 data: List = False
                 ):
        self.data = data
        self.tokenized_data = []
        self.lemmatized_data = []
        self.mfs_data = []
        self.wsd_lesk_applied_data = []
        self.cleaned_data = []
        self.named_entities_data_l = []
        self.tag_conversor = {'NN': 'n', 'NNS': 'n', 'NNP':'n', 'NNPS':'n', 
                              'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 
                              'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ':'v',
                              'RB': 'r', 'RBR': 'r', 'RBS': 'r'}

    def __len__(self):
        return len(self.data)

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
                self.lemmatized_data.append([token.lemma_ for token in doc])
            elif method == 'textserver':
                ts = TextServer('usuari', 'passwd', 'morpho')
                self.lemmatized_data([token[1] for token in ts.morpho(sentence)[0]])
        return self.lemmatized_data

    def most_frequent_synset_data(self, data=False, verbose=True, include_no_pos=False, include_no_cat_synsets=False, count_most_common = False):
        self.mfs_data = []
        t_data = self.data if not data else data
        for sentence in t_data:
            if not isinstance(sentence, list):
                if verbose:
                    print('Applying NLTK tokenization to the sentence')
                sentence = nltk.word_tokenize(sentence)
            self.mfs_data.append(self.most_frequent_synset_sentence(sentence, include_no_pos, include_no_cat_synsets, count_most_common))
        return self.mfs_data

    def most_frequent_synset_sentence(self, sentence, include_no_pos=False, include_no_cat_synsets=False, count_most_common = False):
        sentence = nltk.pos_tag(sentence)
        result_sentence = []
        for (word, tag) in sentence:
            if tag in self.tag_conversor.keys():
                if len(wn.synsets(word, self.tag_conversor[tag])) > 0:
                    result_sentence.append(self.most_frequent_synset(word,self.tag_conversor[tag]) if count_most_common else wn.synsets(word, self.tag_conversor[tag])[0])
                elif include_no_cat_synsets and len(wn.synsets(word)) > 0:
                    result_sentence.append(self.most_frequent_synset(word) if count_most_common else wn.synsets(word)[0])
            elif include_no_pos and len(wn.synsets(word)) > 0:
                result_sentence.append(self.most_frequent_synset(word) if count_most_common else wn.synsets(word)[0])
        return result_sentence

    def wsd_lesk_data(self, data=False, method='nltk', verbose = True, keep_failures = False, synset_word=False):
        self.wsd_lesk_applied_data = []
        t_data = self.data if not data else data
        for sentence in t_data:
            if method == 'nltk':
                if not isinstance(sentence, list):
                    if verbose:
                        print('Applying NLTK tokenization to the sentence')
                    sentence = nltk.word_tokenize(sentence)
                self.wsd_lesk_applied_data.append(self.wsd_lesk_sentence(sentence, keep_failures, synset_word))
        return self.wsd_lesk_applied_data

    def wsd_lesk_sentence(self, sentence, keep_failures = False, synset_word = False):
        disambiguated_sentence =[]
        sentence_tagged = nltk.pos_tag(sentence)
        for (word,tag) in sentence_tagged:
            disambiguated_sentence.append([word, nltk.wsd.lesk(sentence, word, self.tag_conversor[tag] if tag in self.tag_conversor.keys() else None)])
        if synset_word:
            disambiguated_sentence = [[syns[0], syns[1].name().split('.')[0] if syns[1] else syns[1]] for syns in disambiguated_sentence]
        if keep_failures: 
            return [syns[1] if syns[1] else syns[0] for syns in disambiguated_sentence] 
        return [syns[1] for syns in disambiguated_sentence if syns[1]]

    def named_entities_data (self, data=False, method='spacy', verbose = True):
        self.named_entities_data_l = []
        t_data = self.data if not data else data
        for sentence in t_data: 
            self.named_entities_data_l.append(self.named_entities_sentence(sentence, method, verbose))
        return self.named_entities_data_l

    def named_entities_sentence (self, sentence, method = 'nltk', verbose = True):
        if method == 'nltk':
            if not isinstance(sentence, list):
                if verbose:
                    print('Applying NLTK tokenization to the sentence')
                sentence = nltk.word_tokenize(sentence)
            named_entities =  nltk.ne_chunk(nltk.pos_tag(sentence))
            words = []
            last_entity = ''
            entities = []
            for ent in named_entities:
                if isinstance(ent, Tree):
                    if ent.label() == last_entity:
                        entities.append(entities.pop(-1) + ' '+ ' '.join([element[0] for element in ent]))
                    else:
                        entities.append(' '.join([element[0] for element in ent]))
                    last_entity = ent.label()
                else:
                    words.append(ent[0])
                    last_entity = ''
        if method == 'spacy'
            doc = nlp(sentence if not isinstance(sentence, list) else ' '.join(sentence))
            entities=[entity.text for entity in doc.ents]
            not_entities=[word.text for word in doc if not any([ word.text in  entity.text for entity in doc.ents])] 
        return entities+not_entities

    @staticmethod
    def remove_signs(wrd,signs):
        wrd = list(wrd)
        wrd = [word for word in wrd if not any(caracter in signs for caracter in word)]
        wrd = ''.join(wrd)
        return wrd

    @staticmethod
    def most_frequent_synset(word, pos=None):
        syns = wn.synsets(word, pos)
        m = 0
        res = None
        for s in syns:
            for l in s.lemmas():
                if l.count() > m:
                    res = l
        return res.synset()
