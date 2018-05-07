#coding=utf-8
import argparse
import io
import logging
import os

from library.decorators import validate

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


class DynamicFields(object):

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            setattr(self, k, v)

    def __str__(self,):
        return str(self.__dict__)


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
