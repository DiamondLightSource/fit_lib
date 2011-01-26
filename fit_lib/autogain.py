import numpy
import scipy.signal


class Quality:
    '''Abstracts image quality assessment.'''

    # Quality assessment codes returned by image_quality() function
    IMAGE_GOOD = 0
    IMAGE_BRIGHT = 1
    IMAGE_FAINT = 2

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        max = numpy.max(image)
        if max > self.high:
            return self.IMAGE_BRIGHT
        elif max < self.low:
            return self.IMAGE_FAINT
        else:
            return self.IMAGE_GOOD


class StepStrategy:
    def __init__(self, name, min, max):
        self.name = name                # Just for debugging
        self.min = min
        self.max = max
        self.direction = 0
        self.this_step = 1
        self.last_step = 1
        self.max_step = (max - min) // 2

    def __update_step(self, direction):
        if self.direction == direction:
            # New step in same direction as before.  Increment step
            last_step = self.this_step
            self.this_step = min(self.max_step, self.this_step + self.last_step)
            self.last_step = last_step
        else:
            self.direction = direction
            self.this_step = 1
            self.last_step = 1

    def step_up(self, current, max=None):
        if max is None:  max = self.max
        if current < max:
            self.__update_step(+1)
        return min(max, current + self.this_step)

    def step_down(self, current, min=None):
        if min is None:  min = self.min
        if current > min:
            self.__update_step(-1)
        return max(min, current - self.this_step)

    def stand(self):
        self.direction = 0


class AutoGain:
    '''Class to manage camera auto shutter and gain control.'''

    def __init__(self, camera,
            min_gain=None, max_gain=None, min_shutter=None, max_shutter=None,
            nominal_shutter=None):
        # Pick up default gain limits from camera definition
        if min_gain is None:        min_gain = camera.MIN_GAIN
        if max_gain is None:        max_gain = camera.MAX_GAIN
        if min_shutter is None:     min_shutter = camera.MIN_SHUTTER
        if max_shutter is None:     max_shutter = camera.MAX_SHUTTER
        if nominal_shutter is None: nominal_shutter = min_shutter

        # Remember the control parameters
        self.camera = camera
        self.nominal_shutter = nominal_shutter

        # Initialise stepping state: separate stepping state for gain and
        # shutter.
        self.gain    = StepStrategy('gain', min_gain, max_gain)
        self.shutter = StepStrategy('shutter', min_shutter, max_shutter)

    def shutter_down(self, nominal=None):
        self.camera.set_shutter(
            self.shutter.step_down(self.camera.shutter, nominal))
    def shutter_up(self, nominal=None):
        self.camera.set_shutter(
            self.shutter.step_up(self.camera.shutter, nominal))
    def gain_down(self):
        self.camera.set_gain(self.gain.step_down(self.camera.gain))
    def gain_up(self):
        self.camera.set_gain(self.gain.step_up(self.camera.gain))

    def autogain(self, quality):
        '''Steps the camera shutter and gain to optimise the image.'''

        # The gain and shutter control strategy used here is fairly simple.  We
        # have a nominal shutter speed that we'd prefer to use if possible, so
        # we first try moving the shutter to this speed.  Then we try moving the
        # gain until we run out of gain, and then finally the shutter speed is
        # used to complete the range of control.  This results in a trajectory
        # through shutter/gain space somewhat like this:
        #
        #   max_gain ^    +--------+
        #            :    |
        #            :    |
        #            :    |
        #            :    |
        #   min_gain +----+........+
        #   min_shutter   ^        max_shutter
        #             nominal_shutter
        #
        # In the left rectangle we drop the gain or increase the shutter, on the
        # right hand size we drop the shutter or increase the gain.
        gain = self.camera.gain
        shutter = self.camera.shutter
        if quality == Quality.IMAGE_GOOD:
            # Image good, reset both steps to standstill
            self.shutter.stand()
            self.gain.stand()
        elif quality == Quality.IMAGE_BRIGHT:
            # Image too bright, need to step down.
            if shutter > self.nominal_shutter:
                self.shutter_down(self.nominal_shutter)
                self.gain.stand()
            elif gain > self.gain.min:
                self.shutter.stand()
                self.gain_down()
            elif shutter > self.shutter.min:
                self.shutter_down()
                self.gain.stand()
        elif quality == Quality.IMAGE_FAINT:
            # Image too faint, need to step up.
            if shutter < self.nominal_shutter:
                self.shutter_up(self.nominal_shutter)
                self.gain.stand()
            elif gain < self.gain.max:
                self.shutter.stand()
                self.gain_up()
            elif shutter < self.shutter.max:
                self.shutter_up()
                self.gain.stand()


class AutoShutter:
    def __init__(self, camera, gain=None, min_shutter=None, max_shutter=None):
        if min_shutter is None:     min_shutter = camera.MIN_SHUTTER
        if max_shutter is None:     max_shutter = camera.MAX_SHUTTER
        self.camera = camera
        self.shutter = StepStrategy('shutter', min_shutter, max_shutter)
        self.gain = gain

    def autogain(self, quality):
        if self.gain and self.gain != self.camera.gain:
            self.camera.set_gain(self.gain)

        if quality == Quality.IMAGE_GOOD:
            self.shutter.stand()
        elif quality == Quality.IMAGE_BRIGHT:
            self.camera.set_shutter(
                self.shutter.step_down(self.camera.shutter))
        elif quality == Quality.IMAGE_FAINT:
            self.camera.set_shutter(
                self.shutter.step_up(self.camera.shutter))
