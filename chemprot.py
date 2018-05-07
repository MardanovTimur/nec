from corpus import Corpus


class ChemprotCorpus(Corpus):
    doc_pattern = '*_abstracts.tsv'
    ann_pattern = '*_entities.tsv'

    def parse_objects(self):
        # TODO
        pass