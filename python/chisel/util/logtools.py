from functools import wraps
import logging
from datetime import datetime

def timethis(message = None, logit = lambda *args: logging.info(*args), store = lambda _:_):
    """This can decorate functions or methods.
    It times the execution of the decorated object.
    One may specify two arguments: 
    i) a message: defaults to the decorated object's name 
    ii) a logging function: default to logging.info
    Attention: this decorator requires attributes (which may be the default ones),
    therefore don't forget to use it like this @timethis()
    The parentheses are required even though no arguments are given.
    This is just a reminder in case you are unfamiliar with the python syntax for decorators.
    """
    def wrap(f, *args, **kw):
        @wraps(f)
        def wrapped(*args, **kw):
            t0 = datetime.now()
            r = f(*args, **kw)
            delta = datetime.now()-t0
            logit('%s: %s', message if message is not None else f.__name__, delta)
            store(delta)
            return r
        return wrapped
    return wrap

def debugtime(message = None, level = logging.DEBUG, store = lambda _:_):
    """Syntactic sugar for timethis with default logger at DEBUG level"""
    return timethis(message, lambda *args: logging.log(level, *args), store)
