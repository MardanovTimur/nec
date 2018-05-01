import codecs
import os

from library.annotations import parse_brat

file = 'data/corpus_release/data/main/0/0.ann'
for root, dirnames, filenames in os.walk('data/corpus_release/data'):
    for file in filenames:
        if file.find('.ann')!=-1:
            ent, rel = parse_brat(os.path.join(root,file),'utf-8')
            if None in ent or None in rel:
                print('none')
