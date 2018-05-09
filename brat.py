import codecs

from nltk import word_tokenize, re

from corpus import Corpus
from library.annotations import read_doc, get_fictive_relations, get_sentence_in_entities
from models.document import Document
from models.entity import Entity
from models.relation import Relation


class BratCorpus(Corpus):
    doc_pattern = '*[0-9].txt'
    ann_pattern = '*[0-9].ann'

    def parse_objects(self, d_paths, a_paths):
        docs = []
        for path in a_paths:
            text = read_doc(path.replace('ann', 'txt'), self.text_encoding)

            e_list, r_list = parse_brat(path, self.text_encoding)
            kwargs_for_doc = {
                'entities': e_list,
                'relations': r_list,
                'annotation_path': path,
                'text': text,
                'text_path': path.replace('ann','txt'),
            }
            doc = Document(**kwargs_for_doc)
            doc.relations += get_fictive_relations(doc)
            docs.append(doc)
            if len(docs) >= self.train_size:
                break
        return docs


def parse_brat(file_path, encoding):
    entities_list, references_list = [],[]
    file = codecs.open(file_path,'r', encoding=encoding)
    lines = file.readlines()
    text = read_doc(file_path.replace('ann','txt'),encoding)
    doc_id = file_path
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
            references_list.append(Relation(**kwargs_for_relation))

        elif line.startswith('T'):
            fields = re.split("\t+", line)
            id = fields[0].replace('T','')
            entity_type, indexA= fields[1].split(' ')[:2]
            indexB = fields[1].split(' ')[-1]
            value = fields[2].replace('\n', '')
            kwargs_for_entity = {
                'id': int(id),
                'doc_id': doc_id,
                'length': int(indexB) - int(indexA),
                'index_a': int(indexA),
                'index_b': int(indexB),
                'value': value,
                'type': entity_type,
            }
            entities_list.append(Entity(**kwargs_for_entity))

    return entities_list, references_list
