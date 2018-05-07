from corpus import Corpus
from library.annotations import parse_xml
from models.document import Document


class MadeCorpus(Corpus):

    doc_pattern = '*[0-9]_*[0-9]'
    ann_pattern = '*[0-9]_*[0-9].bioc.xml'

    def parse_objects(self):
        docs = []
        for path in self.a_paths:
            e_list, r_list = parse_xml(path, self.encoding)
            kwargs_for_doc = {
                'entities': e_list,
                'relations': r_list,
                'annotation_path': path,
                'text_path': path.replace('annotations', 'corpus').replace('.bioc.xml', ''),
            }
            docs.append(Document(**kwargs_for_doc))
        self.docs = docs
        return docs
