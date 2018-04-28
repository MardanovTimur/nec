#coding=utf-8
from library.lib import DynamicFields

class Reference(DynamicFields):
    # Relation ID
    id = None

    # Type of relation
    type = None

    # Entities ID
    refA = -1
    refB = -1

    # Object of entity
    refAobj = None
    refBobj = None


