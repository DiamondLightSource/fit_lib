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

def prefit_1D_Gaussian(data, origin, scale):
    assert data.ndim == 1
    min = float(data.min())
    max = float(data.max())
    x = numpy.arange(len(data)) - origin
    data_x = data / float(data.sum())
    mean = numpy.sum(x * data_x) * scale
    var  = numpy.sum((x - mean) * data_x) * scale*scale
    return numpy.array([min, max - min, mean, 0.5 / var])

def Gaussian1dValid(params):
    _, _, _, A = params
    return A > 0

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

def WindowGaussian1d(params, window):
    _, _, x_0, A = params
    w = window * math.sqrt(0.5 / A)
    return (x_0 - w, 2*w)

def fit1dGaussian(params, x, data):
    return levmar.fit(
        Gaussian1dValid, Gaussian1d, Gaussian1dJacobian, params, data, (x,))


class Fitter1dGaussian(static.Static):
    prefit = prefit_1D_Gaussian
    fit    = fit1dGaussian
    window = WindowGaussian1d


# ------------------------------------------------------------------------------
# 2D Gaussian fit

# The 2D Gaussian model is parameterised by six parameters:
#
#   base, amplitude, x_0, y_0, A, B, C


def prefit_2D_Gaussian(image, origin, scale):
    '''Computes initial estimates for 2D Gaussian fit to image.  Returns array
    of parameters in the order for fitting.'''

    x0, y0 = origin
    xs, ys = scale
    assert image.ndim == 2 and (numpy.array(image.shape) > 1).all(), \
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
    image_y = image.sum(axis = 0) / total
    # Compute x and y grids with given scale and origin
    x = numpy.arange(len(image_x)) - x0
    y = numpy.arange(len(image_y)) - y0

    # Compute statistics along each axis.
    # Note that these are only good if we have a complete enough curve!
    mean_x = numpy.sum(x * image_x) * xs
    var_x  = numpy.sum((x - mean_x)**2 * image_x) * xs*xs
    mean_y = numpy.sum(y * image_y) * ys
    var_y  = numpy.sum((y - mean_y)**2 * image_y) * ys*ys
    # Convert to initial Gaussian fit parameters
    return numpy.array([
        min, max - min, mean_x, mean_y, 0.5 / var_x, 0.5 / var_y, 0.0])


def WindowGaussian2d(params, window):
    '''Returns a sensible region in which to attempt the fit.  In this case
    we return +-window*sigma around the fitted origin.'''
    _, _, mean_x, mean_y, A, B, _ = params
    win_x = window * math.sqrt(0.5 / A)
    win_y = window * math.sqrt(0.5 / B)
    return ((mean_x - win_x, mean_y - win_y), (2*win_x, 2*win_y))


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

def create_grid(shape):
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


def thin_uniformly(data, factor):
    factor = normalise_sequence(factor, data.ndim)
    return data[tuple(numpy.s_[::f] for f in factor)]

    '''Given a grid and a dataset indexed by that grid reduces the dataset by a
    factor of F^M where M is the number of dimensions of data and F is the
    thinning factor.'''
    # Again numpy index hacks.  The two dimensional expression would be
    #   xy[:, ::factor, ::factor], data[::factor, ::factor]
    ix = tuple(numpy.s_[::f] for f in normalise_sequence(factor, data.ndim))
    return grid[(numpy.s_[:],) + ix], data[ix]


def flatten_grid(grid):
    '''Given an M+1 dimensional grid of shape (M, N_1, ..., N_M) converts it
    into a two dimensional grid of shape (M, N_1*...*N_M).'''
    M = grid.shape[0]
    assert grid.ndim == M + 1, 'Malformed grid'
    return grid.reshape((M, grid.size // M))


def thin_ordered(factor, grid, data):
    '''Thins the data in order of intensity.'''
    thinning = numpy.argsort(data)[::factor]
    return (grid[:, thinning], data[thinning])

def thin_ordered_by(factor):
    return lambda grid, data: thin_ordered(factor, grid, data)


def apply_ROI(data, origin, ROI):
    '''Applies a Region Of Interest to image returning the windowed image
    together with the updated origin.  The ROI is a pair (ROI_origin, extent)
    specifying the offset of the ROI relative to the image origin and the extent
    of the ROI, all units being in pixels.'''
    ROI_origin, extent = ROI
    low = origin + ROI_origin
    high = low + extent
    # Clip the low index to avoid negative indexes, and ensure high is entirely
    # positive for the same reason.
    low[low < 0] = 0
    assert numpy.all(high > 0), 'An axis of the window is outside the dataset'
    index = tuple(numpy.s_[l:h] for l, h in zip(low, high))
    return data[index], origin - low


def apply_scaled_window(data, origin, scaling, window):
    '''Applies the given window to data.'''
    # Convert the computed window into a new Region Of Interest
    window_origin, window_extent = window
    return apply_ROI(data, origin,
        (numpy.int_(window_origin / scaling),
         numpy.int_(window_extent / scaling)))


def gamma_correct(data, gamma, max_data):
    '''Gamma correction.'''
    return data * numpy.exp(gamma * (data / float(max_data) - 1))



# ------------------------------------------------------------------------------


def doFit(fitter, data,
        origin=0, scaling=1,
        ROI=None, window_size=None,
        thinning=None, data_thinning=None,
        gamma=None,
        results=None, **kargs):
    '''General fitter.  Takes the following arguments:

    fitter
        This object should have three attributes, .prefit, .fit and .window:

            initial = .prefit(data, origin, scaling)
                Returns an initial estimate of the fit to data taking into
                account the giving origin and scaling.  The result is an array
                of values suitable for passing to the .fit and .window routines.

            origin, extent = .window(initial, size)
                Returns a "region of interest" computed from the initial
                parameters scaled by size, normally in standard deviations.  The
                region of interest is returned in scaled units, and is converted
                back into pixels to select a sub-region of the data.

            fit, error = .fit(initial, grid, data, **kargs)
                The detailed fitting operation is performed starting from the
                computed initial parameters.  The data will have been thinned
                and windowed by this point, and the grid defines the coordinates
                of each data point with the original origin and scaling taken
                into account.  Extra arguments are passed through to the
                underlying fit algorithm.
                    data is a vector with N points, grid is a MxN matrix where M
                is the dimensionality (.ndim) of the original data set.

    data
        This is the initial data to be fitted, and should have the correct
        number of dimensions expected by the fitter.

    origin, scaling
        The translation from pixel coordinates (indexed by integers starting
        from 0) to underlying model coordinates is specified by these
        parameters, and the returned fit will be in terms of the model
        coordinates.  The translation from pixel coordinates to model
        coordinates is given by the equation

            model_units = scaling * (pixel_units - origin)

        so the origin is in pixel coordinates, and the scaling is in model units
        per pixel.

    ROI
        If specified this is a "Region Of Interest", consisting of a pair
        (ROI_origin, ROI_extent) where ROI_origin is in pixels relative to
        origin and ROI_extent is the size of the region of interest in pixels.

    window_size
        If specified this is a window size to be passed to fitter.window() to
        automatically compute a window on the selected data.

    thinning
        After windowing the data will be thinned by an integer factor.  This can
        be a single integer, or a thinning factor for each dimension of the
        data.

    data_thinning
        Data dependent thinning can also be applied to the data.  This is done
        last, just before gamma correction, so needs to work on the data
        indexing grid.  If specified this must be a function of the form

            grid, data = data_thinning(grid, data)

        where grid and data are as described above for fitter.fit()

    gamma
        If gamma correction on the data is required this should be a pair
        (factor, max_data) where max_data is the maximum normal data value, eg
        255 for 8-bit data, and factor is the required gamma correction.

    results
        If results is passed then intermediate computations will be assigned to
        fields of this structure as follows:

            .grid   Final chosen grid used for fitting
            .data   Final data set used for fitting
    '''

    # Convert inputs into arrays of the appropriate size
    origin = numpy.require(normalise_sequence(origin, data.ndim), dtype=int)
    scaling = numpy.array(normalise_sequence(scaling, data.ndim))

    # Apply Region Of Interest if specified.
    if ROI:
        data, origin = apply_ROI(data, origin, ROI)
    assert data.size, 'No data to fit'

    # Create a sensible initial fit.
    initial = fitter.prefit(data, origin, scaling)

    # Window the data if required.
    if window_size is not None:
        data, origin = apply_scaled_window(
            data, origin, scaling, fitter.window(initial, window_size))

    if thinning is not None:
        # Thin the data uniformly in all dimensions.
        data = thin_uniformly(data, thinning)
        origin /= thinning
        scaling *= thinning

    # Compute the appropriate coordinate grid and perform any further data
    # dependent thinning if required; finally we the data down to a single
    # dimension for fitting.
    grid = flatten_grid(create_grid(data.shape))
    grid = scaling[:, None] * (grid - origin[:, None])
    data = data.flatten()
    if results:
        results.grid = grid
        results.data = data

    # Do data dependent thinning if selected.
    if data_thinning:
        grid, data = data_thinning(grid, data)

    # Finally perform gamma correction on the data before performing the fit.
    if gamma:
        data = gamma_correct(data, *gamma)

    # Perform the fit on the reduced data set and return the result.
    fit, chi2 = fitter.fit(initial, grid, data, **kargs)
    return fit, chi2 / len(data)


def MakeDoFit(fitter):
    return lambda image, **kargs: doFit(fitter, image, **kargs)


doFit2dGaussian   = MakeDoFit(Fitter2dGaussian)
doFit2dGaussian_0 = MakeDoFit(Fitter2dGaussian_0)

doFit1dGaussian   = MakeDoFit(Fitter1dGaussian)
