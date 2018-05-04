from collections import defaultdict

from models.document import Document
from models.entity import Entity
    

def parse_file(basename):
    abstract_fields = ('id', 'title', 'text')
    entity_fields = ('doc_id', 'id', 'type', 'index_a', 'index_b', 'value')

    abstracts_file = open(basename + '_abstracts.tsv', 'rb')
    entities_file = open(basename + '_entities.tsv', 'rb')

    doc_entities = defaultdict(lambda x: [])

    for l in entities_file:
        entity = Entity(**{k: v for k, v in zip(entity_fields, l.split('\t'))})
        doc_entities[entity.doc_id].append(entity)

    for l in abstracts_file:
        doc = Document(**{k: v for k, v in zip(abstract_fields, l.split('\t'))})
        doc.entities = doc_entities[doc.id]
        yield doc


if __name__ == '__main__':
    pass