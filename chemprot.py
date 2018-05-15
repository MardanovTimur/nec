from corpus import Corpus
from parser.ds3 import parse_dataset
import itertools


class ChemprotCorpus(Corpus):
    doc_pattern = '*_abstracts.tsv'
    ann_pattern = '*_entities.tsv'

    def parse_objects(self, d_paths, a_paths):
        basenames = [path.replace('_entities.tsv', '') for path in a_paths]

        docs = itertools.chain(*[
            parse_dataset(basename, encoding=self.text_encoding, with_y=True) for basename in basenames
        ])

        return list(itertools.islice(docs, self.train_size))
