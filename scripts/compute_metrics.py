import math
import os
import numpy as np
from typing import List, Dict, Tuple
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus import wordnet as wn
nltk.download('wordnet_ic')
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')

class ComputeMetrics():
    def __init__(self,
                 data: List = False,
                 metrics: Dict = False,
                 verbose: bool = False
                 ):
        self.data = data
        self.metrics = metrics
        self.verbose = verbose
        self.methods = {'jaccard': self.jaccard_distance,
                        'cosine': self.cosine_distance, 
                        'overlap': self.overlap_distance,
                        'dice': self.dice_distance,
                        'lcs': self.lowest_common_hypernyms_distance,
                        'path': self.path_distance,
                        'wup': self.wup_distance,
                        'lin': self.lin_distance}
        self.synsets_mets = ['lcs', 'path', 'wup', 'lin']
                
    def do(self, save = False):
        results = {}
        metsNames = self.metrics if isinstance(self.metrics, list) else self.metrics.keys()
        for met in metsNames:
            if met in self.synsets_mets and not isinstance(self.data[0][0][0], Synset):
                print(f"{met} cannot be computed because the required data data type is 'Synset'")
                continue
            if self.verbose and self.verbose > 0: print(f'Computing {met}...')
            results[met] = []
            for row in self.data:
                if isinstance(self.metrics, dict):
                    results[met].append(self.methods[met](set(row[0]),set(row[1]),**self.metrics[met]))
                elif isinstance(self.metrics, list):
                    results[met].append(self.methods[met](set(row[0]),set(row[1]),**{}))
    
        if save:
            if not os.path.exists(save):
                os.makedirs(save)
            for metName in metsNames:
                np.save(metName, results[metName])

        return results

    def jaccard_distance(self, sentence1, sentence2):
        return 1 - (len(sentence1.intersection(sentence2))/len(sentence1.union(sentence2)))

    def dice_distance(self, sentence1, sentence2):
        return 1 - ((2*len(sentence1.intersection(sentence2)))/(len(sentence1)+len(sentence2)))

    def overlap_distance(self, sentence1, sentence2):
        return 1 - (len(sentence1.intersection(sentence2))/min(len(sentence1),len(sentence2)))

    def cosine_distance(self, sentence1, sentence2):
        return 1 - (len(sentence1.intersection(sentence2))/math.sqrt(len(sentence1)*len(sentence2)))
    
    def lowest_common_hypernyms_distance(self, sentence1, sentence2):
        return 1 - (sum([1 if self.catch(syns1.lowest_common_hypernyms, syns2, handle=lambda e: False) for syns1 in sentence1 for syns2 in sentence2]) / (len(sentence1)*len(sentence2)))

    def path_distance(self, sentence1, sentence2):
        return 1 - (sum([self.catch(syns1.path_similarity, syns2, handle=lambda e: 0) for syns1 in sentence1 for syns2 in sentence2]) / (len(sentence1)*len(sentence2))

    def wup_distance(self, sentence1, sentence2):
        return 1 - (sum([self.catch(syns1.wup_similarity, syns2, handle=lambda e: 0) for syns1 in sentence1 for syns2 in sentence2]) / (len(sentence1)*len(sentence2))

   def lin_distance(self, sentence1, sentence2):
        return 1 - (sum([self.catch(syns1.lin_similarity, syns2, brown_ic, handle=lambda e: 0) for syns1 in sentence1 for syns2 in sentence2]) / (len(sentence1)*len(sentence2))

    @staticmethod
    def catch(func, *args, handle=lambda e : e, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return handle(e)
