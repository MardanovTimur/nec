#coding=utf-8
import argparse, os, sys, fnmatch, io, re, nltk.sem, logging
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.tag import pos_tag
from library.decorators import validate
from nltk.tokenize import TweetTokenizer
import io, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression

#logger
logger = logging.getLogger('lib.py')

#   Local path of project
PROJECT_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))

#   Path of corpuses
DATA_PATH = os.path.join(PROJECT_PATH, 'data')

#   Word help in --help function
WORD_HELP = u'<surface_all | surface_no_pm | stem | suffix_X>, где в случае surfa\
    ce_all в качестве слов берутся все токены как есть, в случае surface_no_pm - все\
        токены, кроме знаков пунктуаций, в случае stem - \"стемма\" (см. http://snowball.\
        tartarus.org/ ), в случае suffix_X - окончания слов длиной X'

# Define parses for --help functions
PARSER = argparse.ArgumentParser(description=" Project #1 Shevyakov, Mardanov")

# Choice values in parameters for app
WORD_TYPES = (
    'surface_all',
    'surface_no_pm',
    'stem',
    'suffex_X'
)

FEATURES_TYPES = (False, True)

SENTENCE_DIVIDERS = ('.', '!', '?')

# requred variables
REQUIRED = False

# Patterns for files in corpuses and annotations
FILE_PATTERN_IN_CORPUS = {
    u'MADE-1.0': ('*[0-9]_*[0-9]', '*[0-9]_*[0-9].bioc.xml'),
    u'corpus_release': ('*[0-9].txt', '*[0-9].ann'),
    u'chemprot': ('*_abstracts.tsv', '*_entities.tsv')
}


class DynamicFields(object):

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            setattr(self, k, v)

    def __str__(self,):
        return str(self.__dict__)


'''
    return count(int), document_names(list of strings)
'''
@validate
def get_filenames_and_count_of_documents(corpus_path):
    docs, annotations = ([], [])
    re_pattern = FILE_PATTERN_IN_CORPUS.get(corpus_path.split("/")[0])
    if re_pattern is not None:
        for root, dirnames, filenames in os.walk(os.path.join(os.path.abspath(DATA_PATH), corpus_path)):
            for filename in fnmatch.filter(filenames, re_pattern[0]):
                docs.append(os.path.join(root, filename))
            logger.info('get_filenames_and_count_of_documents EXECUTED, {}-documents'.format(len(docs)))
            for filename in fnmatch.filter(filenames, re_pattern[1]):
                annotations.append(os.path.join(root, filename))
            logger.info('get_filenames_and_count_of_documents EXECUTED, {}-annotations'.format(len(annotations)))
        return len(docs), docs, annotations
    else:
        return 0, (), ()



'''
    Functional requirements(HELP for this project)
'''
def parse_args():
    '''
        F1
    '''
    RPARSER = PARSER.add_argument_group('required arguments')
    RPARSER.add_argument('--src-train-texts', type=str,
                         help=u'путь к корпусу, обязательный аргумент,  Данные должны лежать в директории проекта data',
                         metavar='SRC_TRAIN_TEXTS', required=True, nargs='?')
    PARSER.add_argument('--text-encoding', type=str, help=u'кодировка-текста в файлах корпуса',
                        metavar='Encoding', required=False, nargs='?', default="utf-8")
    RPARSER.add_argument('--word-type', type=str, help=WORD_HELP,
                         choices=WORD_TYPES, metavar='Word type', default=WORD_TYPES[0], required=False, nargs='?')
    RPARSER.add_argument('--n', type=int, required=True,
                         help=u'<n-грамность> слова, словосочетания и т.д., обязательный элемент',
                         metavar='NGramm', nargs='?')
    PARSER.add_argument('--features', type=bool, choices=FEATURES_TYPES,
                        help=u'использовать дополнительные hand-crafted признаки, указанные в задании',
                        metavar='Features', default=False, required=False)
    PARSER.add_argument('--laplace', help=u'использовать сглаживание по Лапласу при наличии этого аргумента',
                        default=False,required=False, action='store_true')
    PARSER.add_argument('--unknown-word-freq', type=int,
                        help=u'частота, ниже которой слова в обуч. множестве считаются неизвестными', nargs='?',
                        required=False)
    RPARSER.add_argument('-o', type=str,
                         help=u'путь-куда-сохранить-сериализованную-языковую-модель, обязательный аргумент',
                         metavar='PATH TO SERIALIZE MODEL', required=REQUIRED, nargs='?')
    '''
        F2 and F3
    '''
    PARSER.add_argument('--lm', type=str, help=u'путь к сериализованной языковой модели', required=False)
    RPARSER.add_argument('--src-test-texts', type=str, help=u'путь к тестовой коллекции', required=REQUIRED)
    RPARSER.add_argument('--src-texts', type=str, help=u'путь к коллекции', required=REQUIRED)
    PARSER.add_argument('-o-texts', type=str, help=u'путь для сохранения языковой модели', required=False)

    args = PARSER.parse_args()
    return args


from models.relation import Features
def count_ref_in_document(text, entities, relations, text_path, ref_in, ref_out):
    for rel in relations:
        try:
            # old realisation with indexes (quickly)
            #  ent1, ent2 = sorted(filter(lambda entity: entity.id in (rel.refA, rel.refB), entities), key=lambda x: x.index_a)
            ent1, ent2 = (rel.refAobj, rel.refBobj)
            if len(set(SENTENCE_DIVIDERS).intersection(set(text[ent1.index_b: ent2.index_a])))==0:
                rel.feature_type =  Features.InOneSentence
                ref_in.append(rel)
            else:
                rel.feature_type =  Features.InDifferentSentence
                ref_out.append(rel)
        except:
            '''
                If annotations is not correct.
            '''

'''
    count_in - Count In one sentence
    count_out - Count In different sentences
'''
def references_in_sentence(documents, encoding):
    references_in_one_sentence, references_in_different_sentences = ([], [])
    for document in documents:
        with io.open(document.text_path, encoding='{}'.format(encoding)) as f:
            text = f.read()
            count_ref_in_document(text, document.entities, document.references, document.text_path,
                                  references_in_one_sentence, references_in_different_sentences)
    return references_in_one_sentence, references_in_different_sentences

def count_unique_entites_in_relations(documents):
    s = set()
    for doc in documents:
        for rel in doc.references:
            s.add(rel.refAobj.value)
            s.add(rel.refBobj.value)
    return len(s)

@validate
def statistic_of_corpus(app):
    print 'Count of documents: {}'.format(app.document_count)

    references_count = reduce(lambda initial, y: initial + len(y.references), app.documents, 0)
    app.references_count = references_count
    print 'Count of references [ALL]: {}'.format(references_count)

    references_sentences = references_in_sentence(app.documents, app.text_encoding)
    app.set_refs_in_out(references_sentences[0], references_sentences[1])
    print 'Count of references [IN ONE SENTENSE]: {}\nCount of references [IN DIFERRENT SENTENCES]: {}'.\
            format(len(references_sentences[0]), len(references_sentences[1]))
    del references_sentences

    print 'Count of entities in relations: {}'.format(reduce(lambda c, doc: c+len(doc.references)*2, app.documents, 0))
    print 'Count of UNIQUE entities in relations: {}'.format(
        count_unique_entites_in_relations(app.documents))


from models.pipeline import PipeLine
@validate
def base_line_model(self, entites_test_list):
    left_test, right_test = (dict(entites_test_list).keys(), dict(entites_test_list).values())
    left_words, right_words, types = ([], [], [])
    for document in self.documents:
        for rel in document.references:
            left_words.append(rel.refAobj.value)
            right_words.append(rel.refBobj.value)
            types.append(rel.type)
    pipeline = PipeLine(self, test_counts = 50)
    pipeline.fit(left_words, right_words, types)
    pipeline.transform(left_test, right_test)
    print pipeline.test()

    return pipeline





