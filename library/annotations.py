#coding=utf-8
import xml.etree.ElementTree as ET, io

def parse_xml(file_path,):
    print file_path
    tree = ET.parse(file_path)
    root = tree.getroot()
    print root
    print tree

def convert_to_objects(paths=(), ):
    for path in paths:
        parse_xml(path)

