#coding=utf-8
import logging
from functools import wraps

logger = logging.getLogger('library/decorators.py')

'''
    Decorators for functions
'''

# log-err if function is broken
def validate(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            print e.message
            logger.error('in function = {}'.format(function.__name__))
            logger.error('{}'.format(str(e)))
    return wrapper

