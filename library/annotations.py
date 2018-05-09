#coding=utf-8
import io

from nltk import word_tokenize

from models.relation import Relation


def read_doc(file_path, encoding):
    with io.open(file_path, encoding=encoding) as f:
        return f.read()


def get_sentence_in_entities(text, ent1, ent2):
    if (ent1.index_a > ent2.index_b):
        #Bubble??
        bubble = ent1
        ent1 = ent2
        ent2 = bubble
    indA = ent1.index_a
    indB = ent2.index_b
    text_before_a = text[:indA]
    text_after_b = text[indB:]
    text_before_a_and_after_dot = text_before_a[(None if text_before_a.rfind('.') == -1 else text_before_a.rfind('.') + 1):]
    text_after_b_and_before_dot = text_after_b[:(None if text_after_b.find('.') == -1 else text_after_b.find(('.')) )]
    return " ".join([text_before_a_and_after_dot, text[indA:indB] , text_after_b_and_before_dot])


def get_fictive_relations(doc):
    '''
        Build fictive rels
    '''
    fictive_relations = []

    text = doc.text
    entities = doc.entities
    relations = doc.relations

    # SET IDS of entities, which now in relations
    persistent_ids = map(lambda rel: (rel.refAobj.id, rel.refBobj.id), relations)
    relation_set_by_entity_types = set(relations)

    for i in range(len(entities)-1):
        for j in range(i+1, len(entities)):
            ent1, ent2 = (entities[i], entities[j])
            rel1 = Relation(**{'refAobj': ent1, 'refBobj': ent2, 'is_fictive': True,
                               'text_between': text[ent1.index_b: ent2.index_a],
                               })
            rel2 = Relation(**{'refAobj': ent2, 'refBobj': ent1, 'is_fictive': True,
                               'text_between': text[ent1.index_b: ent2.index_a],
                               })
            filtered_rels = filter(lambda r: r in relation_set_by_entity_types,(rel1, rel2))
            for rel in filtered_rels:
                if (rel.refAobj.id, rel.refBobj.id) not in persistent_ids:
                    rel.text = get_sentence_in_entities(text,rel.refAobj, rel.refBobj,)
                    rel.tokenized_text_between = word_tokenize(rel.text_between)
                    fictive_relations.append(rel)
    return fictive_relations
