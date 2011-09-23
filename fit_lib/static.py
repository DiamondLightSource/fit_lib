# Copyright (c) 2011 Alun Morgan, Michael Abbott, Diamond Light Source Ltd.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
# Contact:
#      Diamond Light Source Ltd,
#      Diamond House,
#      Chilton,
#      Didcot,
#      Oxfordshire,
#      OX11 0DE
#      alun.morgan@diamond.ac.uk, michael.abbott@diamond.ac.uk

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
