import os, sys
from library.lib import parse_args

class App(object):

    def __init__(self, args):
        map(lambda item: setattr(self, item[0], item[1]),dict(filter(lambda x: x[1] is not None,args.__dict__.items())).items())

if __name__ == '__main__':
    args = parse_args()
    app = App(args)

