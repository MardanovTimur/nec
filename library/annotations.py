#coding=utf-8
import io

from models.relation import Relation


def read_doc(file_path, encoding):
    with io.open(file_path, encoding=encoding) as f:
        return f.read()


def get_fictive_relations(entities, relations):
    fictive_relations = []
    # SET IDS of entities, which now in relations
    persistent_ids = map(lambda rel: (rel.refAobj.id, rel.refBobj.id), relations)
    relation_set_by_entity_types = set(relations)
    for i in range(len(entities)-1):
        for j in range(i+1, len(entities)):
            ent1, ent2 = (entities[i], entities[j])
            rel1 = Relation(**{'refAobj': ent1, 'refBobj': ent2, 'is_fictive': True})
            rel2 = Relation(**{'refAobj': ent2, 'refBobj': ent1, 'is_fictive': True})
            filtered_rels = filter(lambda r: r in relation_set_by_entity_types,(rel1, rel2))
            for rel in filtered_rels:
                if (rel.refAobj.id, rel.refBobj.id) not in persistent_ids:
                    fictive_relations.append(rel)
    return fictive_relations
