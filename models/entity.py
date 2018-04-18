#coding=utf-8

class Entity(object):

    id = None
    length = None
    index_a = index_b = -1
    value = None
    type = None

    def __init__(self, *args, **kwargs):
        map(lambda item: setattr(self, item[0], item[1]),dict(filter(lambda x: x[1] is not None, kwargs.items())).items())

    def __str__(self,):
        return str(self.__dict__)

