#coding=utf-8
from library.lib import DynamicFields

class Document(DynamicFields):
    id = None
    annotation_path = None
    text_path = None
    entities = []
    relations = []


