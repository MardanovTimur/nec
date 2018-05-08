from corpus import Corpus
from library.annotations import get_fictive_relations
from library.xmlparser import parse_xml
from models.document import Document


class MadeCorpus(Corpus):

    doc_pattern = '*[0-9]_*[0-9]'
    ann_pattern = '*[0-9]_*[0-9].bioc.xml'

    def parse_objects(self, d_paths, a_paths):
        docs = []
        for path in a_paths:
            e_list, r_list = parse_xml(path, self.text_encoding)
            kwargs_for_doc = {
                'entities': e_list,
                'relations': r_list + get_fictive_relations(e_list, r_list),
                'annotation_path': path,
                'text_path': path.replace('annotations', 'corpus').replace('.bioc.xml', ''),
            }
            docs.append(Document(**kwargs_for_doc))
            if len(docs) >= self.train_size:
                break
        return docs
