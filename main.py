import os, sys
from library.lib import parse_args

class App(object):

    def __init__(self, args):
        print args
        pass

if __name__ == '__main__':
    args = parse_args()
    app = App(args)

