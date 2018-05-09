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

# TODO

def relation_in_one_sentence(self, ):
    '''
    Features:
        Relation in sentence = [
            CPOS(part of speech in relation),
            WVNULL(when no verb in between),
            WVFL(when only verb in between),
            WBNULL(no words in between)],
            WBFL(when only one word in between),
        ]
    '''
    # A
    self.pipeline.ref_in_one_cpos()
    self.pipeline.ref_in_one_wvnull()
    self.pipeline.ref_in_one_wvfl()
    self.pipeline.ref_in_one_wbnull()
    self.pipeline.ref_in_one_wbfl()

    # B
    self.pipeline.init_stanford_dependency_searching()
    try:
        self.pipeline.ref_in_one_dpr2c()
        self.pipeline.ref_in_one_dpr2d()
    except Exception as e:
        print(e)
    finally:
        self.pipeline.dependency_core.close()


def relation_in_different_sentence(self, ):
    self.pipeline.ref_in_diff_sdist()
    self.pipeline.entity_freq_in_doc()
    self.pipeline.whether_type_of_entity_is_unique_in_doc()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    fh = logging.FileHandler('./app.log')
    logger.addHandler(fh)

    args = parse_args()
    corpus = CORPUS_CLS[args.train_path](**args.__dict__)

    #----------------------------------------------------------------------------
    logger.info("First task started : Find relations")
    corpus.first()

    #----------------------------------------------------------------------------
    logger.info("Second task started : Baseline model")

    '''
        Baseline test_data
    '''
    ent2 = ('anesthesia', )
    ent1 = ('ledocaine',)
    corpus.second()

    #----------------------------------------------------------------------------
    logger.info("Third task started : Add extra features for data")
    #---------------------------A-------------------------------------------
    logger.info("Third task started : Extra features for relation in one sentence")
    corpus.third()
    #-----------------------------------------------------------------------
    logger.info('Fourth task started')
    corpus.fourth()
    logger.info('Fourth task finished')

    if args.model_out_path:
        corpus.pipeline.save(args.model_out_path)
