'''Low level camera interface.'''

import cothread
from cothread import catools
import numpy


class Mr1394:
    '''Interface over EPICS to Mr1394 firewire cameras.'''

    MIN_GAIN = 245
    MAX_GAIN = 1023
    MIN_SHUTTER = 1
    MAX_SHUTTER = 4095

    def monitor_variable(self, pv, field):
        def update(value):
            setattr(self, field, value)

        update(catools.caget(pv))
        catools.camonitor(pv, update)

    def __init__(self, name):
        self.name = name
        self.monitor_variable('%s:WIDTH' % name, 'width')
        self.monitor_variable('%s:HEIGHT' % name, 'height')
        self.monitor_variable('%s:STATUS' % name, 'status')
        self.monitor_variable('%s:SET_GAIN' % name, 'gain')
        self.monitor_variable('%s:SET_SHUTTR' % name, 'shutter')
        # Note that the camera provides GAIN and SHUTTER fields, but these don't
        # appear to update normally.

    def get_image(self, timeout=5):
        '''Attempts to retrieve an image with the currently configured height
        and width.  If there's a size mismatch an exception is raised.'''
        raw = catools.caget(
            '%s:DATA' % self.name, timeout=timeout, format=catools.FORMAT_TIME)
        timestamp = raw.timestamp
        # The order of arguments in the reshape reflects the fact that the
        # camera image is formatted line by line.  We transpose the result so
        # that the image can be indexed as image[x,y] with x running
        # horizontally and y vertically without confusion.
        raw = raw.reshape((self.height, self.width)).T
        raw.timestamp = timestamp
        return raw

    def set_gain(self, gain):
        catools.caput('%s:SET_GAIN' % self.name, gain)

    def set_shutter(self, shutter):
        catools.caput('%s:SET_SHUTTR' % self.name, shutter)
