#coding=utf-8
import codecs
import re
import xml.etree.ElementTree as ET, io
from library.decorators import validate
from models.entity import Entity
from models.reference import Reference
from models.document import Document
from nltk.tokenize import word_tokenize
from sets import Set


# Fields realisation for xml childrens
class XML_field(object):

    ref_id = []

    @staticmethod
    def get_params_for_entity(entity):
        xml_obj = XML_field()
        for item in entity._children:
            if item.tag == 'infon':
                xml_obj.type = item.text
            elif item.tag == 'location':
                xml_obj.length = int(item.attrib['length'])
                xml_obj.index = int(item.attrib['offset'])
            elif item.tag == 'text':
                xml_obj.value = item.text
        return xml_obj

    @staticmethod
    def get_params_for_relation(relation):
        xml_obj = XML_field()
        for item in relation._children:
            if item.tag == 'infon':
                xml_obj.type = item.text
            elif item.tag == 'node':
                xml_obj.ref_id.append(int(item.attrib['refid']))
        return xml_obj

def read_doc(file_path, encoding):
    with io.open(file_path, encoding=encoding) as f:
        return f.read()

def get_sentence_in_entities(text, ent1, ent2):
    if (ent1.index_a > ent2.index_b):
        #Bubble??
        buble = ent1
        ent1 = ent2
        ent2 = buble
    indA = ent1.index_a
    indB = ent2.index_b
    text_before_a = text[:indA]
    text_after_b = text[indB:]
    text_before_a_and_after_dot = text_before_a[(None if text_before_a.rfind('.') == -1 else text_before_a.rfind('.') + 1):]
    text_after_b_and_before_dot = text_after_b[:(None if text_after_b.find('.') == -1 else text_after_b.find(('.')) )]
    return " ".join([text_before_a_and_after_dot, text[indA:indB] , text_after_b_and_before_dot])

#MADE-1.0 dataset annotations
def parse_xml(file_path, encoding):

    text = read_doc(file_path.replace('annotations', 'corpus').replace('.bioc.xml', ''), encoding)

    entities_list,references_list = ([], [])
    tree = ET.parse(file_path).getroot()
    entities = tree._children[3]._children[1]._children[1:]
    for entity in entities:
        if entity.tag == "annotation":
            xml_fields_obj = XML_field.get_params_for_entity(entity)
            kwargs_for_entity = {
                'id': int(entity.attrib['id']),
                'length': xml_fields_obj.length,
                'index_a': xml_fields_obj.index,
                'index_b': xml_fields_obj.index + xml_fields_obj.length,
                'value' : xml_fields_obj.value,
                'type': xml_fields_obj.type,
            }
            entities_list.append(Entity(**kwargs_for_entity))
        elif entity.tag == 'relation':
            xml_fields_obj = XML_field.get_params_for_relation(entity)

            entities = [filter(lambda ent: ent.id == xml_fields_obj.ref_id[0], entities_list)[0],
                         filter(lambda ent: ent.id == xml_fields_obj.ref_id[1], entities_list)[0]]
            ent1, ent2 = entities

            _sentence_ = get_sentence_in_entities(text, ent1, ent2)

            kwargs_for_relation = {
                'id': int(entity.attrib['id']),
                'type': xml_fields_obj.type,
                'refA' : xml_fields_obj.ref_id[0],
                'refB' : xml_fields_obj.ref_id[1],
                'refAobj': ent1,
                'refBobj': ent2,
                'text' : _sentence_,
                'text_between': text[ent1.index_b: ent2.index_a],
                'tokenized_text_between': word_tokenize(text[ent1.index_b: ent2.index_a]),
            }
            references_list.append(Reference(**kwargs_for_relation))
            XML_field.ref_id = []
    return entities_list, references_list

#corpuse_release dataset annotations
def parse_brat(file_path, encoding):
    entities_list, references_list = [],[]
    file = codecs.open(file_path,'r', encoding=encoding)
    lines = file.readlines()
    text = read_doc(file_path.replace('ann','txt'),encoding)
    for line in lines:
        if line.startswith('R'):
            id, relation_info = re.split('\t', line)[:2]
            relation_type, arg1, arg2 = relation_info.split(' ')
            refA = arg1.split(':')[1].replace('T','')
            refB = arg2.split(':')[1].replace('T','')
            entities = [filter(lambda ent: ent.id == int(refA), entities_list)[0],
                        filter(lambda ent: ent.id == int(refB), entities_list)[0]]
            ent1, ent2 = entities

            _sentence_ = get_sentence_in_entities(text, ent1, ent2)

            kwargs_for_relation = {
                'id': int(id.replace('R','')),
                'type': relation_type,
                'refA': int(refA),
                'refB': int(refB),
                'refAobj': filter(lambda ent: ent.id == int(refA), entities_list)[0],
                'refBobj': filter(lambda ent: ent.id == int(refB), entities_list)[0],
                'text': _sentence_,
                'text_between': text[ent1.index_b: ent2.index_a],
                'tokenized_text_between': word_tokenize(text[ent1.index_b: ent2.index_a]),
            }
            references_list.append(Reference(**kwargs_for_relation))

        elif line.startswith('T'):
            fields = re.split("\t+", line)
            id = fields[0].replace('T','')
            entity_type, indexA= fields[1].split(' ')[:2]
            indexB = fields[1].split(' ')[-1]
            value = fields[2].replace('\n', '')
            kwargs_for_entity = {
                'id': int(id),
                'length': int(indexB) - int(indexA),
                'index_a': int(indexA),
                'index_b': int(indexB),
                'value': value,
                'type': entity_type,
            }
            entities_list.append(Entity(**kwargs_for_entity))

    return entities_list, references_list


def get_fictive_relations(entities, relations, text_path, encoding):
    fictive_relations = []
    # SET IDS of entities, which now in relations
    persistent_ids = map(lambda rel: (rel.refAobj.id, rel.refBobj.id), relations)
    relation_set_by_entity_types = Set(relations)
    with io.open(text_path, encoding=encoding) as file:
        text = file.read()
    for i in range(len(entities)-1):
        for j in range(i+1, len(entities)):
            ent1, ent2 = (entities[i], entities[j])
            _sentence_1 = get_sentence_in_entities(text,ent1, ent2,)
            rel1 = Reference(**{'refAobj': ent1, 'refBobj': ent2, 'is_fictive': True, 'text': _sentence_1})
            _sentence_2 = get_sentence_in_entities(text,ent2, ent1,)
            rel2 = Reference(**{'refAobj': ent2, 'refBobj': ent1, 'is_fictive': True, 'text': _sentence_2})
            filtered_rels = filter(lambda r: r in relation_set_by_entity_types,(rel1, rel2))
            for rel in filtered_rels:
                if (rel.refAobj.id, rel.refBobj.id) not in persistent_ids:
                    fictive_relations.append(rel)
    return fictive_relations


@validate
def convert_to_objects(a_paths, corpus, encoding, train_size):
    docs = []
    for path in a_paths[:train_size]:
        if ('MADE-1.0' in corpus):
            e_list, r_list = parse_xml(path, encoding)
            kwargs_for_doc = {
                'entities': e_list,
                'references' : r_list,
                'annotation_path': path,
                'text_path': path.replace('annotations', 'corpus').replace('.bioc.xml', ''),
            }
        elif ('corpus_release' in corpus):
            e_list, r_list = parse_brat(path, encoding)
            kwargs_for_doc = {
                'entities': e_list,
                'references': r_list,
                'annotation_path': path,
                'text_path': path.replace('ann','txt'),
            }
        fictive_relations = get_fictive_relations(e_list, r_list, kwargs_for_doc.get('text_path'), encoding)
        kwargs_for_doc.update({'references': r_list + fictive_relations})
        docs.append(Document(**kwargs_for_doc))
    return docs

