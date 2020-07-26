import pandas as pd
import csv

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from gensim.models import FastText

data = pd.read_csv("processed_data/annotatedVec.tsv",sep='\t', quoting=csv.QUOTE_NONE, header=None)
data.columns=['Sent', 'words', 'lang', 'ner-tag']

data = data.fillna(method="ffill")

for index, each in data.iterrows(): 
  if each['words'] in stop_words:
    data.drop(data[data['words'] == each['words']].index, inplace = True)

corpus=[]
for i in data['words'].values:
    corpus.append(str(i).split(" "))
corpus[:1]

model = FastText(corpus, size=100, workers=4,window=5)

model.save('saved_models/fasttext.model')