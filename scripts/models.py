
import os
import sys
import time

from nltk.lm import models
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from typing import List, Dict, Tuple
import string
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import nltk

# NLTK imports
class StatisticalModels():
    def __init__(self,
             data,
             amounts_data,
             test_n,
             models
             ) -> None:
        self.data = data
        self.amount_data = amounts_data
        self.test_n = test_n
        self.models = models if isinstance(models,dict) else {m:[] for m in models}
        self.times_train = {key:[] for key in self.models}
        self.times_test = {key:[] for key in self.models}
        self.times_total = {key:[] for key in self.models}
        self.total_results = {key:[] for key in self.models}
        self.a_models = {'HMM':[nltk.tag.hmm.HiddenMarkovModelTrainer, {},[]],
                         'TnT':[nltk.tag.tnt.TnT, {},[]],
                         'PER':[nltk.tag.perceptron.PerceptronTagger, {'load':False},[]],
                         'CRF':[nltk.tag.CRFTagger, {}, ['crf_tagger_model']]}
        
    def do(self):
        pbar = tqdm(total=100)
        test_data = self.data[self.test_n:]

        for i in tqdm(self.amount_data):
            train_data = self.data[:i]

            for model in self.models.keys():
                time_before_total = time.time()
                time_before_train = time.time()
                mod = self.a_models[model][0](**self.a_models[model][1])
                if model != 'HMM':
                    mod.train(train_data, *self.a_models[model][2])
                else:
                    mod = mod.train_supervised(train_data, *self.a_models[model][2])
                self.times_train[model].append(time.time() - time_before_train)

                time_before_test=time.time()
                self.total_results[model].append(round(mod.accuracy(test_data), 3))
                self.times_test[model].append(time.time() - time_before_test)
                self.times_total[model].append(time.time() - time_before_total)
                
               

            print(i)

        return self.times_train, self.times_test, self.times_total, self.total_results
    
    def results(self):
        df = pd.DataFrame.from_dict(self.total_results)

        for model in self.models:
            plt.plot(self.amount_data, model, data=df, marker='.', markersize=10)

        plt.legend()
        plt.show()

        df_times = pd.DataFrame.from_dict(self.times).round(3)
        df_times['Sentences'] = self.amount_data
        print(df_times)
