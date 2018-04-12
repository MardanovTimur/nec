import os, sys
from library.lib import parse_args

class App(object):

    def __init__(self, args):
        print(args)
        self.lm = args.lm
        self.src_test_texts = args.src_test_texts
        self.src_texts = args.src_texts
        self.o_texts = args.o_texts




if __name__ == '__main__':
    args = parse_args()
    app = App(args)

