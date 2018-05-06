#coding=utf-8
from library.lib import DynamicFields
from enum import Enum


class Features(Enum):
    INIT = 0
    InOneSentence = 1
    InDifferentSentence = 2


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

    # Text between entities
    text_between = ""
    tokenized_text_between = ()

    # Feature type
    feature_type  = Features.INIT

    # Not fictive
    is_fictive = False

    def __eq__(self, o):
        return self.refAobj.type+self.refBobj.type == o.refAobj.type+o.refBobj.type

    def __hash__(self,):
        '''
            For getting uniq types
        '''
        return hash(self.refAobj.type + self.refBobj.type)
