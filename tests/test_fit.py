#!/dls_sw/prod/tools/RHEL5/bin/python2.6

# Simple testing of fit

import optparse
import sys
import os

parser = optparse.OptionParser(
    usage = 'Usage: test_fit [options] <filename>')
parser.add_option(
    '-p', dest = 'plot', default = False, action = 'store_true',
    help = 'Plot results')
parser.add_option(
    '-w', dest = 'window_size', default = 3, type = 'float',
    help = 'Automatic window size')
parser.add_option(
    '-i', dest = 'maxiter', default = 20, type = 'int',
    help = 'Maximum fitting iterations')
parser.add_option(
    '-g', dest = 'gamma', default = 0, type = 'float',
    help = 'Image gamma correction')

options, args = parser.parse_args()
if len(args) != 1:
    parser.print_help()
    sys.exit(1)

filename = ''.join(args)
from pkg_resources import require
require('cothread')
require('scipy')
if options.plot:
    require('matplotlib')

sys.path.append(os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..')))

import scipy.io
from fit_lib import fit_lib


image = scipy.io.loadmat(filename)['image']
print 'Image shape', image.shape

fit, error, results = fit_lib.doFit2dGaussian(
    image, thinning=(5, 5),
    window_size = options.window_size, maxiter = options.maxiter,
    gamma = (options.gamma, 255),
    extra_data = True)
print 'Fitter results'
print 'Baseline ', fit[0]
print 'Peak height ', fit[1]
print 'Origin x (pixels)', fit[2]
print 'Origin y (pixels)', fit[3]
print 'A', fit[4]
print 'B', fit[5]
print 'C ', fit[6]
print 'error', error
print 'suggested region of interest (pixels)'
print 'Origin ', results.origin
print 'Extent ', results.extent
print 'If you want sigmas and rotation then use'
print ' [sigmax, sigmay, rotation] = fit_lib.convert_abc(A, B, C)'

if options.plot:
    import numpy
    import cothread
    cothread.iqt()

    from matplotlib import pyplot

    pyplot.imshow(image.T)
    pyplot.show()

    cothread.WaitForQuit()
