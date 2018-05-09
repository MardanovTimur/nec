import logging

from brat import BratCorpus
from chemprot import ChemprotCorpus
from library.lib import parse_args
from made import MadeCorpus


CORPUS_CLS = {
    u'MADE-1.0': MadeCorpus,
    u'corpus_release': BratCorpus,
    u'chemprot': ChemprotCorpus
}

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    fh = logging.FileHandler('./app.log')
    fh.setFormatter(logging._defaultFormatter)
    logger.addHandler(fh)

    args = parse_args()
    corpus = CORPUS_CLS[args.train_path](**args.__dict__)

    #----------------------------------------------------------------------------
    logger.info("First task started : Find relations")
    corpus.first()

    #----------------------------------------------------------------------------
    # logger.info("Second task started : Baseline model")
    #
    # corpus.second()
    #
    # #----------------------------------------------------------------------------
    # logger.info("Third task started : Add extra features for data")
    # #---------------------------A-------------------------------------------
    # logger.info("Third task started : Extra features for relation in one sentence")
    # corpus.third()
    # #-----------------------------------------------------------------------
    logger.info('Fourth task started')
    corpus.fourth()
    logger.info('Fourth task finished')

    if args.model_out_path:
        corpus.pipeline.save(args.model_out_path)
