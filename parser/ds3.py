import json
from collections import defaultdict

from library.annotations import get_fictive_relations
from models.document import Document
from models.entity import Entity
from models.relation import Relation


def parse_dataset(basename, with_y=False):
    abstract_fields = ('id', 'title', 'text')
    entity_fields = ('doc_id', 'id', 'type', 'index_a', 'index_b', 'value')
    relation_fields = ('doc_id', 'type', 'TODO', '_', 'refA', 'refB')

    def line_to_dict(l, titles):
        d = {k: v.split(':')[-1] for k, v in zip(titles, l.strip().split('\t'))}
        d.pop('_', None)  # remove ignored fields
        return d

    abstracts_file = open(basename + '_abstracts.tsv', 'rb')
    entities_file = open(basename + '_entities.tsv', 'rb')

    doc_entities = defaultdict(lambda: [])
    doc_relations = defaultdict(lambda: [])

    all_entities = dict()

    for l in entities_file:
        entity = Entity(**line_to_dict(l, entity_fields))
        entity.id = entity.doc_id + ':' + entity.id
        entity.index_a = int(entity.index_a)
        entity.index_b = int(entity.index_b)
        all_entities[entity.id] = entity
        doc_entities[entity.doc_id].append(entity)

    if with_y:
        relations_file = open(basename + '_relations.tsv', 'rb')
        for l in relations_file:
            rel = Relation(**line_to_dict(l, relation_fields))
            rel.refA = rel.doc_id + ':' + rel.refA
            rel.refB = rel.doc_id + ':' + rel.refB
            rel.refAobj = all_entities[rel.refA]
            rel.refBobj = all_entities[rel.refB]
            doc_relations[rel.doc_id].append(rel)

    for l in abstracts_file:
        doc = Document(**line_to_dict(l, abstract_fields))
        doc.entities = doc_entities[doc.id]
        doc.relations = doc_relations[doc.id]
        doc.relations += get_fictive_relations(doc)
        yield doc


def export_docs(docs, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join([json.dumps(doc, default=lambda o: o.__dict__) for doc in docs]))


if __name__ == '__main__':
    dev_set = parse_dataset('./data/chemprot/chemprot_development', True)
    export_docs(dev_set, './target/docs.json')
