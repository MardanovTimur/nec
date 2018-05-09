#coding=utf-8
import codecs
import os
import gensim
import Stemmer
import nltk
from gensim.models import Word2Vec, KeyedVectors

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize, pos_tag, RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords, wordnet
from pymystem3 import Mystem



from library.lib import get_filenames_and_count_of_documents


m = Mystem()
stemmer = Stemmer.Stemmer('russian')
def preprocess(text):
    # lemmatized_text = m.lemmatize(text)
    # # text_lemmas = ''.join(lemmatized_text).lower()
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = [w for w in tokens if not w in stopwords.words('russian')]
    return filtered_words



count, files, anns = get_filenames_and_count_of_documents('corpus_release/data')
tokens = []
for f in files:
    text = codecs.open(f, 'r', encoding='utf-8').read()
    tokens.extend(preprocess(text))
    #tokens.extend(word_tokenize(text))

print(len(tokens))
di = {'ADJ':'a', 'ADJ_SAT':'s', 'ADV':'r', 'NOUN':'n', 'VERB':'v'}
s = set(tokens)
l = len(s)
print(l)
list = pos_tag(s,tagset='universal')

lemmatized_tokens = []
i=0
wnl = WordNetLemmatizer()
for token,pos in list:
    p='n'
    if pos in di.keys():
        p = di[pos]
    d =  stemmer.stemWord(token)
    print(d)
    print(i)
    lemmatized_tokens.append(d)
    i+=1
s = set(lemmatized_tokens)
print(len(s))
list = pos_tag(s,tagset='universal')
new_tokens = []
for token, pos in list:
    new_tokens.append(token+'_'+pos)
print('h')
fname = 'ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec'
model = KeyedVectors.load_word2vec_format(fname,binary=False)
c=0
print('here')

print len(set(new_tokens).intersection(set(model.vocab.keys())))

# print('here')
# for token in new_tokens:
#     if token in model.vocab.keys():
#         c+=1
# print(c/l)




