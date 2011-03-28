'''Low level camera interface.'''

import sys
import cothread
from cothread import catools


def format_raw_image(raw_image, width, height):
    '''Reformats a one dimensional array into a two dimensional image while
    preserving the timestamp information.  Will raise an exception if the given
    image dimensions don't match the image size.'''
    # Annoyingly the timestamp gets lost when we reshape the image, so we need
    # to save it and put it back afterwards.
    timestamp = raw_image.timestamp
    # The order of arguments in the reshape reflects the fact that the camera
    # image is formatted line by line.  We transpose the result so that the
    # image can be indexed as image[x,y] with x running horizontally and y
    # vertically without confusion.
    raw_image = raw_image.reshape((height, width)).T
    raw_image.timestamp = timestamp
    return raw_image


class _Subscription:
    '''Used to capture a continuous stream of images.'''

    def __init__(self, camera, pv, max_backlog):
        self.camera = camera
        self.max_backlog = max_backlog
        self.backlog = 0
        self.queue = cothread.EventQueue()
        self.subscription = catools.camonitor(pv, self.__on_update,
            format = catools.FORMAT_TIME)

    def close(self):
        self.queue.close()
        self.subscription.close()

    def __on_update(self, value):
        self.backlog += 1
        if self.max_backlog and self.backlog > self.max_backlog:
            # Whoops, too many unprocessed values.  Discard this one and close
            # everything.
            print >>sys.stderr, 'Image subscription backlog too large'
            self.close()
        else:
            # We store the current camera image dimensions with the image as we
            # capture it so we have a fighting chance of getting the right
            # dimensions if the image changes during capture.
            value = value[:self.camera.width * self.camera.height]
            self.queue.Signal((value, self.camera.width, self.camera.height))

    def get_image(self, timeout=5):
        raw_image, width, height = self.queue.Wait(timeout)
        self.backlog -= 1
        return format_raw_image(raw_image, width, height)


    # The following two methods turn this subscription object into an iterable
    # object.
    def __iter__(self): return self
    next = get_image


class Mr1394:
    '''Interface over EPICS to Mr1394 firewire cameras.'''

    MIN_GAIN = 245
    MAX_GAIN = 1023
    MIN_SHUTTER = 1
    MAX_SHUTTER = 4095

    def monitor_variable(self, pv, field):
        def update(value):
            setattr(self, field, int(value))

        update(catools.caget(pv))
        catools.camonitor(pv, update)

    def __init__(self, name):
        self.name = name
        self.monitor_variable('%s:WIDTH' % name[0:14], 'width')
        self.monitor_variable('%s:HEIGHT' % name[0:14], 'height')
        self.monitor_variable('%s:STATUS' % name, 'status')
        self.monitor_variable('%s:SET_GAIN' % name, 'gain')
        self.monitor_variable('%s:SET_SHUTTR' % name, 'shutter')
        # Note that the camera provides GAIN and SHUTTER fields, but these don't
        # appear to update normally.

    def get_image(self, timeout=5):
        '''Attempts to retrieve an image with the currently configured height
        and width.  If there's a size mismatch an exception is raised.'''
        raw_image = catools.caget('%s:DATA' % self.name,
            count = self.width * self.height,
            timeout = timeout, format = catools.FORMAT_TIME)
        return format_raw_image(raw_image, self.width, self.height)

    def set_gain(self, gain):
        catools.caput('%s:SET_GAIN' % self.name, gain)

    def set_shutter(self, shutter):
        catools.caput('%s:SET_SHUTTR' % self.name, shutter)

    def subscribe(self, max_backlog = 10):
        return _Subscription(self, '%s:DATA' % self.name, max_backlog)


class TomGigE:
    """Interface over EPICS to Tom's GigE cameras cameras."""

    MIN_GAIN = 245
    MAX_GAIN = 1023
    MIN_SHUTTER = 1
    MAX_SHUTTER = 4095

    def monitor_variable(self, pv, field):
        def update(value):
            setattr(self, field, int(value))

        update(catools.caget(pv))
        catools.camonitor(pv, update)

    def __init__(self, name):
        self.name = name
        self.monitor_variable('%s:CAM:ArraySizeX_RBV' % name, 'width')
        self.monitor_variable('%s:CAM:ArraySizeY_RBV' % name, 'height')
        self.monitor_variable('%s:CAM:DetectorState_RBV' % name, 'status')
        self.monitor_variable('%s:CAM:Gain_RBV' % name, 'gain')
        self.monitor_variable('%s:CAM:AcquireTime_RBV' % name, 'shutter')
        # Note that the camera provides GAIN and SHUTTER fields, but these don't
        # appear to update normally.

    def get_image(self, timeout=10):
        '''Attempts to retrieve an image with the currently configured height
        and width.  If there's a size mismatch an exception is raised.'''
        raw_image = catools.caget('%s:ARR:ArrayData' % self.name,
            count = self.width * self.height,
            timeout = timeout, format = catools.FORMAT_TIME)
        return camera.format_raw_image(raw_image, self.width, self.height)

    def set_gain(self, gain):
        catools.caput('%s:CAM:Gain' % self.name, gain)

    def set_shutter(self, shutter):
        catools.caput('%s:CAM:AcquireTime' % self.name, shutter)

    def subscribe(self, max_backlog = 10):
        return _Subscription(self, '%s:ARR:ArrayData' % self.name, max_backlog)

