import math
import os
import numpy as np
from typing import List, Dict, Tuple

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
                        'dice': self.dice_distance}
                
    def do(self, save = False):
        results = {}
        metsNames = self.metrics if isinstance(self.metrics, list) else self.metrics.keys()
        for met in metsNames:
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
