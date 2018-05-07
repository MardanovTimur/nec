#coding=utf-8
import codecs
import re
import xml.etree.ElementTree as ET, io
from library.decorators import validate
from models.entity import Entity
from models.relation import Relation
from models.document import Document
from nltk.tokenize import word_tokenize


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


# MADE-1.0 dataset annotations
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
            if (xml_fields_obj.ref_id[0] > xml_fields_obj.ref_id[1]):
                entities.reverse()
            ent1, ent2 = entities

            kwargs_for_relation = {
                'id': int(entity.attrib['id']),
                'type': xml_fields_obj.type,
                'refA' : xml_fields_obj.ref_id[0],
                'refB' : xml_fields_obj.ref_id[1],
                'refAobj': ent1,
                'refBobj': ent2,
                'text_between': text[ent1.index_b: ent2.index_a],
                'tokenized_text_between': word_tokenize(text[ent1.index_b: ent2.index_a]),
            }
            references_list.append(Relation(**kwargs_for_relation))
            XML_field.ref_id = []
    return entities_list, references_list


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
            kwargs_for_relation = {
                'id': int(id.replace('R','')),
                'type': relation_type,
                'refA': int(refA),
                'refB': int(refB),
                'refAobj': filter(lambda ent: ent.id == int(refA), entities_list)[0],
                'refBobj': filter(lambda ent: ent.id == int(refB), entities_list)[0],
                'text_between': text[ent1.index_b: ent2.index_a],
                'tokenized_text_between': word_tokenize(text[ent1.index_b: ent2.index_a]),
            }
            references_list.append(Relation(**kwargs_for_relation))

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
