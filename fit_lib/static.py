'''A simple class for defining classes for gathering static methods.'''

import types

class Static(object):
    '''This is designed to be used for classes which gather constants and are
    never intended to be instantiated.  Unfortunately any functions assigned
    when creating such a class are converted to bound methods, which isn't what
    we want in such an application.  This class carefully undoes this by
    converting any functions to static methods before the class is created.'''

    class __StaticMeta(type):
        def __new__(cls, name, bases, dict):
            for n, v in dict.items():
                if isinstance(v, types.FunctionType):
                    dict[n] = staticmethod(v)
            return type.__new__(cls, name, bases, dict)

    __metaclass__ = __StaticMeta

    def __new__(self, *argv, **argk):
        assert False, 'Cannot instantiate Static class'
