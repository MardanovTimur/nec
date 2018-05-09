from xml.etree import ElementTree as ET

from nltk import word_tokenize

from library.annotations import read_doc, get_sentence_in_entities
from models.entity import Entity
from models.relation import Relation


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


# MADE-1.0 dataset annotations
def parse_xml(file_path, encoding):

    text = read_doc(file_path.replace('annotations', 'corpus').replace('.bioc.xml', ''), encoding)

    doc_id = file_path

    entities_list,references_list = ([], [])
    tree = ET.parse(file_path).getroot()
    entities = tree._children[3]._children[1]._children[1:]
    for entity in entities:
        if entity.tag == "annotation":
            xml_fields_obj = XML_field.get_params_for_entity(entity)
            kwargs_for_entity = {
                'id': int(entity.attrib['id']),
                'doc_id': doc_id,
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

            _sentence_ = get_sentence_in_entities(text, ent1, ent2)

            kwargs_for_relation = {
                'id': int(entity.attrib['id']),
                'type': xml_fields_obj.type,
                'refA' : xml_fields_obj.ref_id[0],
                'refB' : xml_fields_obj.ref_id[1],
                'refAobj': ent1,
                'refBobj': ent2,
                'text': _sentence_,
                'text_between': text[ent1.index_b: ent2.index_a],
                'tokenized_text_between': word_tokenize(text[ent1.index_b: ent2.index_a]),
            }
            references_list.append(Relation(**kwargs_for_relation))
            XML_field.ref_id = []
    return entities_list, references_list, text