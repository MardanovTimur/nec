import argparse

PARSER = argparse.ArgumentParser()

def parse_args():
    PARSER.add_argument('-a','--argument', help='Help for this argument')

    args = PARSER.parse_args()
    return args

