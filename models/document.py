#coding=utf-8
from library.lib import DynamicFields

class Document(DynamicFields):
    annotation_path = None
    text_path = None
    entities = []
    references = []

    def __str__(self,):
        return str(self.__dict__)

