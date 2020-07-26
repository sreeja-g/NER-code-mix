import pandas as pd
import csv 

class SentenceGetter(object):
            
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(),
                                                           s["ner-tag"].values.tolist())]
        self.grouped = self.data.groupby("Sent").apply(agg_func)
        self.sentences = [s for s in self.grouped][1:]
    
    def get_next(self):
        try:
            s = self.grouped["sent: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

data = pd.read_csv("processed_data/annotatedVec.tsv",sep='\t', quoting=csv.QUOTE_NONE, header=None)
data.columns=['Sent', 'words', 'lang', 'ner-tag']

getter = SentenceGetter(data)
sentences = getter.sentences

import pickle5 as pickle

pickle_out = open("sentence_data/train_data.pickle","wb")
pickle.dump(sentences[:int(len(sentences)*0.7)], pickle_out)
pickle_out.close()

pickle_out = open("sentence_data/test_data.pickle","wb")
pickle.dump(sentences[int(len(sentences)*0.7):], pickle_out)
pickle_out.close()
