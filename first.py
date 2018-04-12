#coding=utf-8
import os, sys, fnmatch, io, re, nltk.sem
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk

_mathces = []
IN = re.compile(r'.*\bin\b(?!\b.+ing)')

def count_of_documents(file_name='MADE-1.0'):
    for root, dirnames, filenames in os.walk(os.path.abspath(os.path.join(os.path.abspath(__file__),'..','corpuses',))):
        for filename in fnmatch.filter(filenames, '*[0-9]'):
            _mathces.append(os.path.join(root, filename))
    return len(_mathces)

def find_relation():
    for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
        for rel in nltk.sem.relextract.extract_rels('ORG', 'LOC', doc,corpus='ieer', pattern = IN):
             print (nltk.sem.relextract.rtuple(rel))

def extract_entities(_mathces):
    entities = {}
    for file in _mathces:
        with io.open(file, 'r') as f:
            for sentence in sent_tokenize(f.read()):
                chunks = ne_chunk(pos_tag(word_tokenize(sentence), lang='ru'))
                entities.get(file, []).extend([chunk for chunk in chunks if hasattr(chunk, 'label')]) #label = entity
    return entities

if __name__ == '__main__':
    print count_of_documents()
    print extract_entities(_mathces)
