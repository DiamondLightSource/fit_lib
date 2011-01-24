'''Low level camera interface.'''

from cothread import catools
import numpy
import scipy.signal


def monitor_variable(self, pv, field, default=None):
    def update(value):
        setattr(self, field, value)
    setattr(self, field, default)
    catools.camonitor(pv, update)


class Mr1394:
    '''Interface over EPICS to Mr1394 firewire cameras.'''

    MIN_GAIN = 245
    MAX_GAIN = 1023
    MIN_SHUTTER = 1
    MAX_SHUTTER = 4095

    def __init__(self, name):
        self.name = name
        monitor_variable(self, '%s:WIDTH' % name, 'width', 0)
        monitor_variable(self, '%s:HEIGHT' % name, 'height', 0)
        monitor_variable(self, '%s:STATUS' % name, 'status', 0)
        monitor_variable(self, '%s:SET_GAIN' % name, 'gain', 0)
        monitor_variable(self, '%s:SET_SHUTTR' % name, 'shutter', 0)
        # Note that the camera provides GAIN and SHUTTER fields, but these don't
        # appear to update normally.

    def get_image(self, timeout=5):
        '''Attempts to retrieve an image with the currently configured height
        and width.  If there's a size mismatch an exception is raised.'''
        raw = catools.caget('%s:DATA' % self.name, timeout=timeout)
        # The order of arguments in the reshape reflects the fact that the
        # camera image is formatted line by line.  We transpose the result so
        # that the image can be indexed as image[x,y] with x running
        # horizontally and y vertically without confusion.
        return raw.reshape((self.height, self.width)).T

    def set_gain(self, gain):
        catools.caput('%s:SET_GAIN' % self.name, gain)

    def set_shutter(self, shutter):
        catools.caput('%s:SET_SHUTTR' % self.name, shutter)
