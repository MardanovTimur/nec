#coding=utf-8
import argparse


WORD_HELP = u'<surface_all | surface_no_pm | stem | suffix_X>, где в случае surfa\
    ce_all в качестве слов берутся все токены как есть, в случае surface_no_pm – все\
        токены, кроме знаков пунктуаций, в случае stem – “стемма” (см. http://snowball.\
        tartarus.org/ ), в случае suffix_X – окончания слов длиной X'

PARSER = argparse.ArgumentParser(description="Shevyakov, Mardanov")

WORD_TYPES = ('surface_all', 'surface_no_pm', 'stem', 'suffex_X')
FEATURES_TYPES = (False, True)

REQUIRED = False

'''
    Functional requirements
'''
def parse_args():
    '''
        F1
    '''
    RPARSER = PARSER.add_argument_group('required arguments')
    RPARSER.add_argument('--src-train-texts', type=str,help=u'путь к корпусу, обязательный аргумент',
                        metavar='SRC_TRAIN_TEXTS', required=REQUIRED, nargs='?')
    PARSER.add_argument('--text-encoding', type=str,help=u'кодировка-текста в файлах корпуса',
                        metavar='Encoding', required=False, nargs='?')
    RPARSER.add_argument('--word-type', type=str,help=WORD_HELP,
                        choices=WORD_TYPES,metavar='Word type',default=WORD_TYPES[0], required=False, nargs='?')
    RPARSER.add_argument('-n', type=int, required=REQUIRED, help= u'<n-грамность> слова, словосочетания и т.д., обязательный элемент',
                        metavar='NGramm', nargs='?')
    PARSER.add_argument('--features', type=bool,  choices=FEATURES_TYPES, help=u'использовать дополнительные hand-crafted признаки, указанные в задании',
                        metavar='Features', default=False, required=False)
    PARSER.add_argument('--laplace' ,help=u'использовать сглаживание по Лапласу при наличии этого аргумента', nargs='?', required=False)
    PARSER.add_argument('--unknown-word-freq', help=u'частота, ниже которой слова в обуч. множестве считаются неизвестными', nargs='?', required=False)
    RPARSER.add_argument('-o', type=str,help=u'путь-куда-сохранить-сериализованную-языковую-модель, обязательный аргумент',
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
