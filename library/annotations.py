#coding=utf-8
import io
import xml.etree.ElementTree as ET

from nltk.tokenize import word_tokenize
from sets import Set

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

def get_fictive_relations(entities, relations):
    fictive_relations = []
    # SET IDS of entities, which now in relations
    persistent_ids = map(lambda rel: (rel.refAobj.id, rel.refBobj.id), relations)
    relation_set_by_entity_types = Set(relations)
    for i in range(len(entities)-1):
        for j in range(i+1, len(entities)):
            ent1, ent2 = (entities[i], entities[j])
            rel1 = Reference(**{'refAobj': ent1, 'refBobj': ent2, 'is_fictive': True})
            rel2 = Reference(**{'refAobj': ent2, 'refBobj': ent1, 'is_fictive': True})
            filtered_rels = filter(lambda r: r in relation_set_by_entity_types,(rel1, rel2))
            for rel in filtered_rels:
                if (rel.refAobj.id, rel.refBobj.id) not in persistent_ids:
                    fictive_relations.append(rel)
    return fictive_relations
