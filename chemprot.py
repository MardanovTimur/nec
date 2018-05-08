from corpus import Corpus
from parser.ds3 import parse_dataset


class ChemprotCorpus(Corpus):
    doc_pattern = '*_abstracts.tsv'
    ann_pattern = '*_entities.tsv'

    def parse_objects(self, d_paths, a_paths):
        basenames = [path.replace('_entities.tsv', '') for path in a_paths]

        docs = []

        for basename in basenames:
            docs += list(parse_dataset(basename, True))

        return docs
