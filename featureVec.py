import pandas as pd
import csv
import numpy as np

class SentenceGetter(object):
            
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(),
                                                           s["ner-tag"].values.tolist())]
        self.grouped = self.data.groupby("Sent").apply(agg_func)
        self.sentences = [s for s in self.grouped][1:]
        
        sentence_text_agg_func = lambda s: ' '.join(s["words"].values.tolist())
        self.grouped_sentence_text = self.data.groupby("Sent").apply(sentence_text_agg_func)
        self.sentence_text = [s for s in self.grouped_sentence_text][1:]
    
    def get_next(self):
        try:
            s = self.grouped["sent: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def mean_normalize(num,max_,min_):

    return 2*((num-min_)/(max_-min_))-1        


def numericFeatures():

    data = pd.read_csv("processed_data/annotatedVec.tsv",sep='\t', quoting=csv.QUOTE_NONE, header=None)
    data.columns=['Sent', 'words', 'lang', 'ner-tag']

    data = data.fillna(method="ffill")

    words = list(set(data["words"].values))
    words.append("ENDPAD")
    tags = list(set(data["ner-tag"].values))

    word2idx = {w: mean_normalize(i,len(words)-1,0) for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    word2Suff2idx = {w[-2:]: mean_normalize(i,len(words)-1,0) for i, w in enumerate(words)}
    word3Suff2idx = {w[-3:]: mean_normalize(i,len(words)-1,0) for i, w in enumerate(words)}
    wordLower2idx = {w.lower(): mean_normalize(i,len(words)-1,0) for i, w in enumerate(words)}
    binaryIdx = {"True": 1, "False": 0}

    getter = SentenceGetter(data)

    sentences = getter.sentences

    def word2features(sent, i):
        word = sent[i][0] 

        features = {
            'bias': 1.0,
            'word': word2idx[word],
            'word.lower()': wordLower2idx[word.lower()],
            'word[-3:]': word3Suff2idx[word[-3:]],
            'word[-2:]': word2Suff2idx[word[-2:]],
            'word.isupper()': binaryIdx[str(word.isupper())],
            'word.istitle()': binaryIdx[str(word.istitle())],
            'word.isdigit()': binaryIdx[str(word.isdigit())],
            'word.startsWith#()': binaryIdx[str(word.startswith("#"))],
            'word.startsWith@()': binaryIdx[str(word.startswith("@"))],
            'word.1stUpper()': binaryIdx[str(word[0].isupper())],
            'word.isAlpha()': binaryIdx[str(word.isalpha())],
            'word.Tag': tag2idx[sent[i][1]],
        }

        if i > 0:
            word1 = sent[i-1][0]
            features.update({
                '-1:word': word2idx[word1],
                '-1:word.lower()': wordLower2idx[word1.lower()],
                '-1:word.istitle()': binaryIdx[str(word1.istitle())],
                '-1:word.isupper()': binaryIdx[str(word1.isupper())],
                '-1:word.istitle()': binaryIdx[str(word1.istitle())],
                '-1:word.isdigit()': binaryIdx[str(word1.isdigit())],
                '-1:word.startsWith#()': binaryIdx[str(word1.startswith("#"))],
                '-1:word.startsWith@()': binaryIdx[str(word1.startswith("@"))],
                '-1:word.1stUpper()': binaryIdx[str(word1[0].isupper())],
                '-1:word.isAlpha()': binaryIdx[str(word1.isalpha())],
            })
        else:
            features['BOS'] = binaryIdx[str("True")]

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word': word2idx[word1],
                '+1:word.lower()': wordLower2idx[word1.lower()],
                '+1:word.istitle()': binaryIdx[str(word1.istitle())],
                '+1:word.isupper()': binaryIdx[str(word1.isupper())],
                '+1:word.istitle()': binaryIdx[str(word1.istitle())],
                '+1:word.isdigit()': binaryIdx[str(word1.isdigit())],
                '+1:word.startsWith#()': binaryIdx[str(word1.startswith("#"))],
                '+1:word.startsWith@()': binaryIdx[str(word1.startswith("@"))],
                '+1:word.1stUpper()': binaryIdx[str(word1[0].isupper())],
                '+1:word.isAlpha()': binaryIdx[str(word1.isalpha())],
            })
        else:
            features['EOS'] = binaryIdx[str("True")]

        return features
        
    def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]

    def sent2labels(sent):
        return [label for token, label in sent]

    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]

    csv_columns = ['+1:word', '+1:word.1stUpper()', '+1:word.isAlpha()', '+1:word.isdigit()', '+1:word.istitle()','+1:word.isupper()', '+1:word.lower()', '+1:word.startsWith#()', '+1:word.startsWith@()', 'BOS', '-1:word', '-1:word.1stUpper()', '-1:word.isAlpha()', '-1:word.isdigit()', '-1:word.istitle()', '-1:word.isupper()','-1:word.lower()', '-1:word.startsWith#()', '-1:word.startsWith@()', 'EOS', 'bias', 'word', 'word.1stUpper()', 'word.isAlpha()', 'word.isdigit()', 'word.istitle()','word.isupper()', 'word.lower()', 'word.startsWith#()', 'word.startsWith@()', 'word[-2:]', 'word[-3:]', 'word.Tag']

    with open('processed_data/featureVec.csv', 'w') as ofile:
        writer = csv.DictWriter(ofile, csv_columns)
        writer.writeheader()
        for s in X:

            for d in s:
            
                writer.writerow(d)  

    X_train = [[{key:val for key, val in each.items() if key != 'word.Tag'} for each in sent2features(s)] for s in sentences[:int(len(sentences)*0.7)]]
    y_train = [sent2labels(s) for s in sentences[:int(len(sentences)*0.7)]]

    X_test = [[{key:val for key, val in each.items() if key != 'word.Tag'} for each in sent2features(s)] for s in sentences[int(len(sentences)*0.7):]]
    y_test = [sent2labels(s) for s in sentences[int(len(sentences)*0.7):]]


    return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test, 'sentences':getter.sentence_text}



