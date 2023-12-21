import os
import io
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, path):
        self.path = path
        self.files = [file for file in os.listdir(path) if 'STS' in file]
        self.inputs = {file.split('.')[-2]:file for file in self.files if file.split('.')[1] == 'input'}
        self.labels = {file.split('.')[-2]:file for file in self.files if file.split('.')[1] == 'gs'}
    
    def __getitem__(self,dataName = 'all'):
        dataname  = ['SMTeuroparl','MSRvid', 'MSRpar'] if dataName == 'all' else dataName 
        dataname = dataname if isinstance(dataname, list) else [dataname]
        return pd.concat([self.read_as_df(self.inputs[name], self.labels[name]) for name in dataname], ignore_index=True)

    def read_as_df(self, data, labels):
        try:
            data =  pd.read_csv(self.path + data, sep='\t', header=None)
            data['gs'] = pd.read_csv(self.path + labels, sep='\t', header = None)
        except:
            with open(self.path + data) as f:
                lines = f.readlines()
            for index in range(len(lines)):
                lines[index] = lines[index].replace('\"', ' ')
            data = pd.read_csv(io.StringIO(''.join(lines)), sep='\t',header=None, on_bad_lines='warn')
            data['gs'] = pd.read_csv(self.path + labels,sep='\t',header=None)

        return data

def pretty_table(dct, name):
    print(F'\n--- Results for {name}')
    table = PrettyTable(dct.keys())
    for row in zip(*[dct[k] for k in dct]):
      table.add_row(row)
    print(table)


class ShowResults():
    def __init__(self,
                 results, 
                 groups,
                 show = True):
        self.mets_results = results.copy()
        self.gs = self.mets_results['gs']
        del self.mets_results['gs']
        self.groups = groups if isinstance(groups, list) else list(groups.values())
        self.metrics = list(self.mets_results.keys())
        self.types = results[self.metrics[0]].keys()
        self.name_groups = groups if isinstance(groups, list) else list(groups.keys())
        if show:
            self.do()
        self.create_dataframe()

    def __getitem__(self, metrics):
        metrics = metrics if isinstance(metrics, list) else [metrics] 
        results = {k:self.dict_group(v,metrics) for k,v  in zip(self.name_groups, self.groups)}
        for name, dic in results.items():
            pretty_table(dic, name)

    def do(self):
        results = {k:self.dict_group(v,self.metrics) for k,v  in zip(self.name_groups, self.groups)}
        for name, dic in results.items():
            pretty_table(dic, name)

    def pearson_group(self, metric, group):
        return [round(pearsonr(self.gs, 1-self.mets_results[metric][key])[0], 3) for key in self.types if group == str(key).split('_')[-1]]

    def names_group(self, group):
        return [' '.join(key.split('_')).upper() for key in self.types if group == str(key).split('_')[-1]]
    
    def dict_group(self, group, metrics):
        dg = {'Category': self.names_group(group)}
        for metric in metrics:
            dg[metric] = self.pearson_group(metric,group)
        return dg
    
    def create_dataframe(self, groups=None, metrics=None):
        groups = groups if groups else self.groups
        metrics = metrics if metrics else self.metrics
        self.dataframe = pd.concat([pd.DataFrame(self.dict_group(v,metrics)) for v  in self.groups])
        return self.dataframe
    
    def heatmap (self): 
        sns.heatmap(self.dataframe.set_index('Category'), annot=True, linewidths=0.3,cmap='YlGnBu',yticklabels=True,fmt='.3f', vmin=0, vmax=1)
        plt.gcf().set_size_inches(5,12)
        plt.show()
        