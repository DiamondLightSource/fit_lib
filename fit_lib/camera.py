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

    def monitor_variable(self, pv, field, default, wait=False):
        def update(value):
            setattr(self, field, value)
            if wait:
                event.Signal()
        if wait:
            event = cothread.Event(auto_reset = False)
            self.__waits.append(event)
        setattr(self, field, default)
        catools.camonitor(pv, update)

    def __init__(self, name):
        self.name = name
        self.__waits = []
        self.monitor_variable('%s:WIDTH' % name, 'width', 0, True)
        self.monitor_variable('%s:HEIGHT' % name, 'height', 0, True)
        self.monitor_variable('%s:STATUS' % name, 'status', 0)
        self.monitor_variable('%s:SET_GAIN' % name, 'gain', 0)
        self.monitor_variable('%s:SET_SHUTTR' % name, 'shutter', 0)
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

    def wait_start(self, timeout=5):
        '''Waits for the important field to be populated.'''
        cothread.WaitForAll(self.__waits, timeout = timeout)
