#coding=utf-8
import argparse

PARSER = argparse.ArgumentParser()


def parse_args():
    PARSER.add_argument('--lm', type=str, help=u'путь к сериализованной языковой модели', required=False)
    PARSER.add_argument('--src-test-texts', type=str, help=u'путь к тестовой коллекции', required=True)
    PARSER.add_argument('--src-texts', type=str, help=u'путь к коллекции', required=True)
    PARSER.add_argument('-o-texts', type=str, help=u'путь для сохранения языковой модели', required=False)

    args = PARSER.parse_args()
    return args
