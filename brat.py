from corpus import Corpus
from library.annotations import parse_brat
from models.document import Document


class BratCorpus(Corpus):
    doc_pattern = '*[0-9].txt'
    ann_pattern = '*[0-9].ann'

    def parse_objects(self):
        docs = []
        for path in self.a_paths:
            e_list, r_list = parse_brat(path, self.encoding)
            kwargs_for_doc = {
                'entities': e_list,
                'relations': r_list,
                'annotation_path': path,
                'text_path': path.replace('ann','txt'),
            }
            docs.append(Document(**kwargs_for_doc))
        self.docs = docs
        return docs
