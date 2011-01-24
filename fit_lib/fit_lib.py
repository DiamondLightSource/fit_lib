'''Support for fitting a 2D Guassian to an image.'''

import math
import numpy
import types

import levmar
import static



# ------------------------------------------------------------------------------
# Ellipse coordinate conversion functions.

# The two functions here convert between two canonical representations of an
# ellipse centred at the origin:
#   1. Major axis, minor axis, angle of major axis
#   2. Coefficents of polynomial A x^2 + B y^2 + C x y
# It is easier and more stable to fit to a Gaussian in form (2), but we want the
# answer in form (1), hence these conversion functions.

def convert_abc(A, B, C):
    '''Converts ellipse parameters A, B, C for an ellipse of the form
           2      2
        A x  + B y  + C x y
    to major axis, minor axis squared and tilt angle.'''

    eps_skew = 1e-9
    eps_round= 1e-9
    if abs(C) < eps_skew * abs(A - B):
        # Skew is very small i.e. beam is horizontal or vertical
        t = 0.5 * C / (A - B)
    elif abs(C) < eps_round and abs(A - B) < eps_round:
        # Round horizontal beam.  In this case t becomes unconstrained, so force
        # it to zero.
        t = 0
    else:
        surd = math.sqrt((A - B)**2 + C**2)
        if A < B:
            # Take sign(surd) = sign(A - B) so that |t| <= 1, so
            #   -pi/4 <= theta <= pi/4
            surd = - surd
        t = (-A + B + surd) / C

    theta = math.atan(t)
    sigma_x = 1. / math.sqrt(2*A + C * t)
    sigma_y = 1. / math.sqrt(2*B - C * t)
    return sigma_x, sigma_y, -theta * 180. / math.pi


def convert_sigma_theta(sigma_x, sigma_y, theta):
    '''Converts from major, minor, theta form into A, B, C form of ellipse.'''
    ct = numpy.cos(-theta)
    st = numpy.sin(-theta)
    isx = 0.5 / sigma_x**2
    isy = 0.5 / sigma_y**2

    A = isx * ct**2 + isy * st**2
    B = isx * st**2 + isy * ct**2
    C = 2 * ct * st * (isx - isy)
    return A, B, C


# ------------------------------------------------------------------------------
# 1D Gaussian fit

# The 1D Gaussian model is parameterised by four parameters:
#
#   base, amplitude, x_0, A

def Gaussian1dValid(params):
    return params[-1] > 0

def Gaussian1d(params, x):
    g_0, K, x_0, A = params
    x = x - x_0
    return g_0 + K * numpy.exp(-(A * x**2))

def Gaussian1dJacobian(params, x):
    g_0, K, x_0, A = params
    x = x - x_0
    x2 = x * x
    E = numpy.exp(-(A * x2))
    KE = K * E
    return numpy.array([
        numpy.ones(len(E)),             # dG/dg_0
        E,                              # dG/dK
        2 * A * x * KE,                 # dG/dx_0
        - x2 * KE])                     # dG/dA

def fit1dGaussian(params, x, data):
    return levmar.fit(
        Gaussian1dValid, Gaussian1d, Gaussian1dJacobian, params, data, (x,))


# ------------------------------------------------------------------------------
# 2D Gaussian fit

# The 2D Gaussian model is parameterised by six parameters:
#
#   base, amplitude, x_0, y_0, A, B, C


def prefit_2D_Gaussian(image):
    '''Computes initial estimates for 2D Gaussian fit to image.  Returns array
    of parameters in the order for fitting.'''

    assert len(image.shape) == 2 and (numpy.array(image.shape) > 1).all(), \
        'Can only fit to rectangular image'

    # This is done by projecting the image onto X and Y (by taking means) and
    # then computing statistics from these projections.  The results are the
    # combined into an initial estimate for the 2D fit.

    # Estimate vertical range
    min = float(image.min())
    max = float(image.max())
    # Project the image onto its axes, convert these into densities
    total = float(image.sum())
    image_x = image.sum(axis = 1) / total
    x = numpy.arange(len(image_x))
    image_y = image.sum(axis = 0) / total
    y = numpy.arange(len(image_y))

    # Compute statistics along each axis.
    # Note that these are only good if we have a complete enough curve!
    mean_x = numpy.sum(x * image_x)
    var_x  = numpy.sum((x - mean_x)**2 * image_x)
    mean_y = numpy.sum(y * image_y)
    var_y  = numpy.sum((y - mean_y)**2 * image_y)
    # Convert to initial Gaussian fit parameters
    return numpy.array([
        min, max - min, mean_x, mean_y, 0.5 / var_x, 0.5 / var_y, 0.0])


def WindowGaussian2d(params, window):
    '''Returns a sensible region in which to attempt the fit.'''
    _, _, mean_x, mean_y, A, B, _ = params
    win_x = window * math.sqrt(0.5 / A)
    win_y = window * math.sqrt(0.5 / B)
    return (
        (max(0, int(mean_x - win_x)), int(mean_x + win_x)),
        (max(0, int(mean_y - win_y)), int(mean_y + win_y)))


def Gaussian2dValid(params):
    A, B, C = params[-3:]
    return A > 0 and B > 0 and 4 * A * B > C * C

def Gaussian2d(params, xy):
    '''Test function used to compute modelled Gaussian on the given x,y
    vectors.'''
    g_0, K, x_0, y_0, A, B, C = params
    x, y = xy
    x = x - x_0
    y = y - y_0
    return g_0 + K * numpy.exp(-(A * x**2 + B * y**2 + C * x*y))

def Gaussian2dJacobian(params, xy):
    g_0, K, x_0, y_0, A, B, C = params
    x, y = xy
    x = x - x_0
    y = y - y_0
    x2 = x * x
    y2 = y * y
    xy = x * y
    E = numpy.exp(-(A * x2 + B * y2 + C * xy))
    KE = K * E
    return numpy.array([
        numpy.ones(len(E)),             # dG/dg_0
        E,                              # dG/dK
        (2 * A * x + C * y) * KE,       # dG/dx_0
        (2 * B * y + C * x) * KE,       # dG/dy_0
        - x2 * KE,                      # dG/dA
        - y2 * KE,                      # dG/dB
        - xy * KE])                     # dG/dC


def Gaussian2d_0(params, xy, g_0):
    '''Modified Guassian calculation with zero baseline.'''
    return Gaussian2d(numpy.concatenate(([g_0], params)), xy)

def Gaussian2dJacobian_0(params, xy, g_0):
    '''Modified Jacobian calculation with zero baseline.'''
    return Gaussian2dJacobian(numpy.concatenate(([g_0], params)), xy)[1:]


def fit2dGaussian(params, xy, data, **kargs):
    '''Given a good initial estimate and flattenned and thinned data returns the
    best 2D Gaussian fit to the dataset.'''
    return levmar.fit(
        Gaussian2dValid, Gaussian2d, Gaussian2dJacobian,
        params, data, (xy,), **kargs)

def fit2dGaussian_0(params, xy, data, **kargs):
    '''A modification of fit2dGaussian which forces the baseline to a constant
    value.  Still takes and returns the same parameter set, but allows no
    variation of g_0.'''
    g_0 = params[0]
    result, chi2 = levmar.fit(
        Gaussian2dValid, Gaussian2d_0, Gaussian2dJacobian_0,
        params[1:], data, (xy, g_0), **kargs)
    return numpy.concatenate(([g_0], result)), chi2


# Gather the key elements of these fitters.

class Fitter2dGaussian(static.Static):
    prefit = prefit_2D_Gaussian
    fit    = fit2dGaussian
    window = WindowGaussian2d

class Fitter2dGaussian_0(static.Static):
    prefit = prefit_2D_Gaussian
    fit    = fit2dGaussian_0
    window = WindowGaussian2d


# ------------------------------------------------------------------------------

# Windowing functionality.

def grid(shape):
    '''Given a shape tuple (N_1, ..., N_M) returns a coordinate grid of shape
        (M, N_1, ..., N_M)
    with g[m, n_1, ..., n_M] = n_(m+1).  This can be used as an index into an
    array of the given shape by converting the grid to a tuple, and indeed
        a[tuple(grid(a.shape))] = a
    in general.'''
    # Some rather obscure numpy index trickery.  We could write the expression
    # below more simply as
    #   numpy.mgrid[:shape[0], :shape[1]]
    # except that the form below will work for any length of shape.  The mgrid
    # construction takes an index expression and turns it into a grid which
    # cycles over all the points in the index.
    return numpy.mgrid[tuple(numpy.s_[:n] for n in shape)]


def normalise_sequence(input, rank):
    '''Lifted from numpy.  Converts input to a list of length rank, replicating
    as necessary if it's not already a sequence.'''
    if (isinstance(input, (types.IntType, types.LongType, types.FloatType))):
        return [input] * rank
    else:
        return input


def thin_uniformly(grid, data, factor):
    '''Given a grid and a dataset indexed by that grid reduces the dataset by a
    factor of F^M where M is the number of dimensions of data and F is the
    thinning factor.'''
    # Again numpy index hacks.  The two dimensional expression would be
    #   xy[:, ::factor, ::factor], data[::factor, ::factor]
    ix = tuple(numpy.s_[::f] for f in normalise_sequence(factor, data.ndim))
    return grid[(numpy.s_[:],) + ix], data[ix]


def apply_window(window, grid, data):
    '''Applies the given window to xy and data.'''
    ix = tuple(numpy.s_[l:h] for l, h in window)
    return grid[(numpy.s_[:],) + ix], data[ix]


def flatten_grid(grid):
    '''Given an M+1 dimensional grid of shape (M, N_1, ..., N_M) converts it
    into a two dimensional grid of shape (M, N_1*...*N_M).'''
    M = grid.shape[0]
    assert grid.ndim == M + 1, 'Malformed grid'
    return grid.reshape((M, grid.size // M))


def thin_ordered(factor, xy, data):
    '''Thins the data in order of intensity.'''
    thinning = numpy.argsort(data)[::factor]
    return (xy[:, thinning], data[thinning])


def apply_ROI(image, ROI):
    '''Applies an optional Region Of Interest to image returning the windowed
    image (if appropriate) together with the associated offsets.  The ROI is
    either None if not required or four numbers definining the region of
    interest: (min_x, max_x, min_y, max_y).'''
    if ROI:
        min_x, max_x, min_y, max_y = ROI
        image = image[min_x:max_x, min_y:max_y]
        if image.size:
            return image, min_x, min_y
        else:
            raise FitError('Region of interest lies outside image')
    else:
        return image, 0, 0


def gamma_correct(data, gamma, max_data):
    '''Gamma correction.'''
    return data * numpy.exp(gamma * (data / float(max_data) - 1))



# ------------------------------------------------------------------------------


def doFit(fitter, data, thinning=None, gamma=None, window=0, ROI=None, **kargs):
    '''General fitter.  Takes the following arguments:

    fitter
        This object should have two attributes, .prefit and .fit.  The routine
        .prefit(data) returns a set of initial parameters from the given data,
        and .fit(initial, xy, data) performs the fit on decimated data.

    data
        This is the initial data to be fitted.
    '''

    # Start with a sensible initial guess.  This will guide our subsequent
    # windowing and thinning.  Create a full grid to cover the selected data.
    initial = fitter.prefit(data)
    xy_grid = grid(data.shape)

    # Window the data if required.  We window with both an optional "Region Of
    # Interest" and an automatic window determined from the prefit parameters.
    if window > 0:
        xy_grid, data = apply_window(
            fitter.window(initial, window), xy_grid, data)

    if thinning:
        # Thin the data on the original grid so we thin uniformly in both
        # dimensions.
        xy_grid, data = thin_uniformly(xy_grid, data, thinning)

    # Flatten the data down to a single dimension for fitting.
    xy = flatten_grid(xy_grid)
    data = data.flatten()

    # Perform gamma correction on the data before performing the fit but after
    # everything else, simply to reduce the cost of this operation.
    if gamma:
        data = gamma_correct(data, *gamma)

    # Perform the fit on the reduced data set and return the result.
    result, chi2 = fitter.fit(initial, xy, data, **kargs)
    return result, chi2 / len(data)

def MakeDoFit(fitter):
    return lambda image, **kargs: doFit(fitter, image, **kargs)


doFit2dGaussian   = MakeDoFit(Fitter2dGaussian)
doFit2dGaussian_0 = MakeDoFit(Fitter2dGaussian_0)
