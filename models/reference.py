#coding=utf-8

class Reference(object):
    id = None
    refA = refB = -1

    def __init__(self, *args, **kwargs):
        map(lambda item: setattr(self, item[0], item[1]),dict(filter(lambda x: x[1] is not None, kwargs.items())).items())

    def __str__(self,):
        return str(self.__dict__)

