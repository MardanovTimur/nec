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


def log_time(func, cls_name=None):
    logger = logging.getLogger('log_time')
    name = func.__name__
    if cls_name is not None:
        name = '{}.{}'.format(cls_name, name)

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug('{} start'.format(name))
        r = func(*args, **kwargs)
        logger.debug('{} end'.format(name))
        return r
    return wrapper