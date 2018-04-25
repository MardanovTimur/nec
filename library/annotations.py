#coding=utf-8
import xml.etree.ElementTree as ET, io
from library.decorators import validate
from models.entity import Entity
from models.reference import Reference
from models.document import Document


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


#MADE-1.0 dataset annotations
def parse_xml(file_path,):
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
            kwargs_for_relation = {
                'id': int(entity.attrib['id']),
                'refA' : xml_fields_obj.ref_id[0],
                'refB' : xml_fields_obj.ref_id[1],
            }
            references_list.append(Reference(**kwargs_for_relation))
            XML_field.ref_id = []
    return entities_list, references_list

@validate
def convert_to_objects(a_paths= ()):
    docs = []
    for path in a_paths:
        e_list, r_list = parse_xml(path)
        kwargs_for_doc = {
            'entities': e_list,
            'references' : r_list,
            'annotation_path': path,
            'text_path': path.replace('annotations', 'corpus').replace('.bioc.xml', ''),
        }
        docs.append(Document(**kwargs_for_doc))
    return docs

