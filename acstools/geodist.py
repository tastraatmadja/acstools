__version__ = '0.0.1'
__date__    = '09-Nov-2023'
__author__  = 'Tri L. Astraatmadja'

from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from scipy import signal
from scipy.spatial import distance


def get_rectangle_midpoint(coords):
    '''
    Given 4 points of a rectangle vertices, calculate the middle point of the rectangle.

    Parameters:
    - coords (numpy.ndarray): A 2D NumPy array representing the coordinates of the rectangle vertices.
                             Each row corresponds to the (x, y) coordinates of a vertex.

    Returns:
    - float: The x-coordinate of the midpoint of the rectangle.

    Example:
    >>> import numpy as np
    >>> rectangle_coords = np.array([[0, 0], [2, 0], [2, 3], [0, 3]])
    >>> get_rectangle_midpoint(rectangle_coords)
    1.0
    '''
    midpoint1 = 0.5 * (coords + np.roll(coords, -1, axis=0))
    midpoint2 = 0.5 * (midpoint1 + np.roll(midpoint1, -2, axis=0))
    return midpoint2[0]

def gaussian_kernel_2d(x, xi, y, yi, hx, hy):
    """
    Compute the value of a 2D Gaussian kernel at a given point.

    Parameters:
    - x (float): X-coordinate of the point where the kernel is evaluated.
    - xi (float): X-coordinate of the kernel center.
    - y (float): Y-coordinate of the point where the kernel is evaluated.
    - yi (float): Y-coordinate of the kernel center.
    - hx (float): Bandwidth or standard deviation along the x-axis.
    - hy (float): Bandwidth or standard deviation along the y-axis.

    Returns:
    - float: The value of the 2D Gaussian kernel at the specified point.

    Note:
    The formula for the 2D Gaussian kernel is given by:
    K(x, y; xi, yi, hx, hy) = exp(-0.5 * ((x - xi)/hx)^2 + ((y - yi)/hy)^2)

    Example:
    >>> import numpy as np
    >>> x, xi, y, yi, hx, hy = 1.0, 0.0, 1.0, 0.0, 0.5, 1.0
    >>> gaussian_kernel_2d(x, xi, y, yi, hx, hy)
    0.8824969025845955
    """
    dx = (x - xi) / hx
    dy = (y - yi) / hy
    u2 = dx ** 2 + dy ** 2

    del dx
    del dy

    return np.exp(-0.5 * u2)


def get_KDE_image(x, y, xBins, yBins, hx, hy):
    """
    Generate a 2D Kernel Density Estimate (KDE) image from given data points.

    Parameters:
    - x (numpy.ndarray): Array of x-coordinates of data points.
    - y (numpy.ndarray): Array of y-coordinates of data points.
    - xBins (numpy.ndarray): Array defining the bins along the x-axis for the KDE.
    - yBins (numpy.ndarray): Array defining the bins along the y-axis for the KDE.
    - hx (float): Bandwidth or standard deviation along the x-axis for the Gaussian kernel.
    - hy (float): Bandwidth or standard deviation along the y-axis for the Gaussian kernel.

    Returns:
    - numpy.ndarray: Normalized 2D KDE image.

    Notes:
    - The function uses a 2D Gaussian kernel for each data point to generate the KDE.
    - The KDE image is computed on a grid defined by xBins and yBins.

    Example:
    >>> import numpy as np
    >>> x_data = np.random.rand(100)
    >>> y_data = np.random.rand(100)
    >>> x_bins = np.linspace(0, 1, 20)
    >>> y_bins = np.linspace(0, 1, 20)
    >>> hx_value, hy_value = 0.1, 0.1
    >>> kde_image = get_KDE_image(x_data, y_data, x_bins, y_bins, hx_value, hy_value)
    """
    idx0 = np.digitize(x, xBins)
    idx1 = np.digitize(y, yBins)

    xMids = 0.5 * (xBins[idx0 - 1] + xBins[idx0])
    yMids = 0.5 * (yBins[idx1 - 1] + yBins[idx1])

    weights = gaussian_kernel_2d(x, xMids, y, yMids, hx, hy)

    nx, ny = xBins.size - 1, yBins.size - 1

    kdeImage = np.zeros((ny, nx))

    for i, j, value in zip(idx1 - 1, idx0 - 1, weights):
        ## print(i, j, value)
        kdeImage[i, j] += value

    normalizedKDEImage = kdeImage / kdeImage.sum()

    del idx0
    del idx1
    del xMids
    del yMids
    del weights
    del kdeImage

    return normalizedKDEImage


def get_normal_triad(c):
    '''
    Given a SkyCoord coordinate, return the normal triad pqr at the coordinate. First row = p, second row = q, third row = r

    Parameters:
    - c (astropy.coordinates.SkyCoord): Celestial coordinates object representing right ascension and declination.

    Returns:
    - numpy.ndarray: 3x3 matrix representing the normal triad in the celestial coordinate system.

    Notes:
    - The normal triad matrix is a 3x3 matrix that defines the local coordinate system
      aligned with the celestial coordinate system at a given point on the sky.
    - The input celestial coordinates (c) should be provided using the astropy.coordinates.SkyCoord class.

    Example:
    >>> from astropy.coordinates import SkyCoord
    >>> import numpy as np
    >>> celestial_position = SkyCoord(ra=10.0, dec=20.0, unit='deg')
    >>> normal_triad_matrix = get_normal_triad(celestial_position)
    '''
    sinRA = np.sin(c.ra)
    cosRA = np.cos(c.ra)
    sinDE = np.sin(c.dec)
    cosDE = np.cos(c.dec)

    return np.array([[-sinRA, cosRA, 0.0],
                     [-sinDE * cosRA, -sinDE * sinRA, cosDE],
                     [+cosDE * cosRA, cosDE * sinRA, sinDE]])


def get_r_triad(c):
    '''
    Given a SkyCoord object, return the radial vector, i.e. the unit vector along the line of sight toward the object

    Parameters:
    - c (astropy.coordinates.SkyCoord): Celestial coordinates object representing right ascension and declination.

    Returns:
    - numpy.ndarray: 3D vector representing the radial triad in the celestial coordinate system.

    Notes:
    - The radial triad vector points along the line of sight from the observer to the celestial object.
    - The input celestial coordinates (c) should be provided using the astropy.coordinates.SkyCoord class.

    Example:
    >>> from astropy.coordinates import SkyCoord
    >>> import numpy as np
    >>> celestial_position = SkyCoord(ra=10.0, dec=20.0, unit='deg')
    >>> radial_triad_vector = get_r_triad(celestial_position)
    '''
    sinRA = np.sin(c.ra)
    cosRA = np.cos(c.ra)
    sinDE = np.sin(c.dec)
    cosDE = np.cos(c.dec)

    return np.vstack([+cosDE * cosRA, cosDE * sinRA, sinDE]).transpose()


def get_normal_coordinates(c, pqr0):
    '''
    Given a SkyCoord object and a normal triad pqr0 around a reference point, return the normal coordinates

    Parameters:
    - c (astropy.coordinates.SkyCoord): Celestial coordinates object representing right ascension and declination.
    - pqr0 (numpy.ndarray): 3D vector in the radial triad at a reference position.

    Returns:
    - tuple: A tuple containing two astropy.units.Quantity objects representing the normal coordinates (xi, eta) in arcseconds.

    Notes:
    - The normal coordinates (xi, eta) represent the projected angular displacement in the plane perpendicular to the line of sight.
    - The input celestial coordinates (c) should be provided using the astropy.coordinates.SkyCoord class.
    - The input vector pqr0 should be in the radial triad coordinate system.

    Example:
    >>> from astropy.coordinates import SkyCoord
    >>> import numpy as np
    >>> import astropy.units as u
    >>> celestial_position = SkyCoord(ra=10.0, dec=20.0, unit='deg')
    >>> reference_vector = np.array([1.0, 0.0, 0.0])
    >>> xi, eta = get_normal_coordinates(celestial_position, reference_vector)
    '''
    r = get_r_triad(c)
    r0DotR = np.sum(pqr0[2] * r, axis=1)

    ## Normal coordinates in arcsec
    xi = ((np.sum(pqr0[0] * r, axis=1) / r0DotR) * u.radian).to(u.arcsec)
    eta = ((np.sum(pqr0[1] * r, axis=1) / r0DotR) * u.radian).to(u.arcsec)

    del r
    del r0DotR

    return xi, eta


def phase_correlate_2d(x1, y1, x2, y2, dx, dy, hx, hy):
    '''
    Calculate phase correlation between two sets of 2D data points to determine relative shift.

    Parameters:
    - x1 (numpy.ndarray): X-coordinates of the first dataset.
    - y1 (numpy.ndarray): Y-coordinates of the first dataset.
    - x2 (numpy.ndarray): X-coordinates of the second dataset.
    - y2 (numpy.ndarray): Y-coordinates of the second dataset.
    - dx (float): Grid spacing along the x-axis for the phase correlation.
    - dy (float): Grid spacing along the y-axis for the phase correlation.
    - hx (float): Bandwidth or standard deviation along the x-axis for the Gaussian kernel.
    - hy (float): Bandwidth or standard deviation along the y-axis for the Gaussian kernel.

    Returns:
    - tuple: A tuple containing the estimated shift along the x-axis (shiftX),
             the estimated shift along the y-axis (shiftY), the peak index (idx),
             and the phase correlation result (corrIm).

    Notes:
    - The phase correlation method is used to determine the relative shift between two images.
    - It involves calculating the 2D Kernel Density Estimate (KDE) images for both datasets,
      performing a phase correlation, and finding the peak index to estimate the shift.
    - Grid spacing (dx and dy) defines the resolution for phase correlation.
    - Bandwidth (hx and hy) determines the kernel width for KDE.

    Example:
    >>> import numpy as np
    >>> x1_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y1_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> x2_data = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
    >>> y2_data = np.array([1.1, 2.2, 3.0, 4.0, 5.0])
    >>> dx_value, dy_value, hx_value, hy_value = 0.1, 0.1, 0.2, 0.2
    >>> shiftX, shiftY, idx, corrIm = phase_correlate_2d(x1_data, y1_data, x2_data, y2_data, dx_value, dy_value, hx_value, hy_value)
    '''
    ## Calculate coverage of the
    xMin, xMax = np.floor(np.nanmin(np.hstack([x1, x2]))), np.ceil(np.nanmax(np.hstack([x1, x2])))
    yMin, yMax = np.floor(np.nanmin(np.hstack([y1, y2]))), np.ceil(np.nanmax(np.hstack([y1, y2])))

    nPointsX = int((xMax - xMin) // dx) + 1
    nPointsY = int((yMax - yMin) // dy) + 1

    nBinsX = nPointsX + 1
    nBinsY = nPointsY + 1

    print(xMin, xMax, yMin, yMax, nPointsX, nPointsY, nBinsX, nBinsY)

    xBins = np.linspace(xMin - 0.5 * dx, xMax + 0.5 * dx, nBinsX, endpoint=True)
    yBins = np.linspace(yMin - 0.5 * dy, yMax + 0.5 * dy, nBinsY, endpoint=True)

    kdeIm1 = get_KDE_image(x1, y1, xBins, yBins, hx, hy)
    kdeIm2 = get_KDE_image(x2, y2, xBins, yBins, hx, hy)

    corrIm = signal.correlate(kdeIm1, kdeIm2, mode='full', method='auto')

    del kdeIm1
    del kdeIm2

    idx = np.unravel_index(np.argmax(corrIm), corrIm.shape)

    shiftX = (dx * (idx[1] - int(corrIm.shape[1] // 2)))
    shiftY = (dy * (idx[0] - int(corrIm.shape[0] // 2)))

    return shiftX, shiftY, idx, corrIm


def get_upper_triangular_matrix_number_of_elements(n):
    """
    Calculate the number of elements in an upper triangular matrix of size n x n.

    Parameters:
    - n (int): The size (number of rows or columns) of the square matrix.

    Returns:
    - int: The total number of elements in the upper triangular matrix.

    Notes:
    - An upper triangular matrix is a square matrix in which all elements below the main diagonal are zero.
    - The function computes the number of non-zero elements in the upper triangular matrix based on its size.
    - This number is used to determine the amount of storage or processing required for such matrices.

    Example:
    >>> n = 4
    >>> num_elements = get_upper_triangular_matrix_number_of_elements(n)
    >>> print(num_elements)  # Output: 10
    """
    return int(n * (n + 1) / 2)


def get_upper_triangular_index(i, j):
    '''
    Return the upper triangular index of a square matrix with any size
    Example below for n = 5, the p index for any given i,j index will
    be as the following:
    * i\j|  0    1    2    3    4 |
    * ---|-------------------------
    *  0 |  0 |  1 |  3 |  6 | 10 |
    *    |-------------------------
    *  1 |  2 |  4 |  7 | 11 |
    *    |--------------------
    *  2 |  5 |  8 | 12 |
    *    |---------------
    *  3 |  9 | 13 |
    *    |----------
    *  4 | 14 |
    * ---------
     Parameters:
        - i (int): The row index of the element.
        - j (int): The column index of the element.

    Returns:
        - int: The index of the element within the upper triangular matrix.

    Example:
    >>> row_index = 2
    >>> col_index = 1
    >>> element_index = get_upper_triangular_index(row_index, col_index)
    >>> print(element_index)  # Output: 8
    '''
    return int(get_upper_triangular_matrix_number_of_elements((i+j)) + i)


def get_SIP_model(X, Y, pOrder, scalerX=1.0, scalerY=1.0):
    """
    Generate the Simple Imaging Polynomial (SIP) model matrix and scaling factors for given input coordinates.

    Parameters:
    - X (numpy.ndarray): Array of x-coordinates.
    - Y (numpy.ndarray): Array of y-coordinates.
    - pOrder (int): The polynomial order for the SIP model.
    - scalerX (float): Scaling factor for the x-coordinates (default is 1.0).
    - scalerY (float): Scaling factor for the y-coordinates (default is 1.0).

    Returns:
    - tuple: A tuple containing the SIP model matrix (XModel) and the flattened scaling factor array (scaler).

    Notes:
    - The SIP model is used for geometric distortion correction in astronomical imaging.
    - The function generates a matrix (XModel) that represents the polynomial model for the given input coordinates.
    - The scaling factors (scalerX and scalerY) allow for coordinate scaling if needed.
    - The matrix XModel is used to compute the corrected coordinates based on the distortion model.
    - The polynomial order (pOrder) defines the complexity of the distortion model.

    Example:
    >>> import numpy as np
    >>> X_coords = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> Y_coords = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> polynomial_order = 2
    >>> scale_factor_X = 0.5
    >>> scale_factor_Y = 0.7
    >>> sip_model_matrix, scaling_factors = get_SIP_model(X_coords, Y_coords, polynomial_order, scale_factor_X, scale_factor_Y)
    """
    n = pOrder + 1

    vanderX = np.vander(X / scalerX, n, increasing=True)
    vanderY = np.vander(Y / scalerY, n, increasing=True)

    vanderScalerX = np.vander(np.array([scalerX]), n, increasing=True)
    vanderScalerY = np.vander(np.array([scalerY]), n, increasing=True)

    P = get_upper_triangular_matrix_number_of_elements(n)  ## Number of parameters PER AXIS!

    XModel = np.zeros((X.shape[0], P))
    scaler = np.zeros((1, P))

    for ii in range(n):
        for jj in range(n - ii):
            pVal = vanderX[:, ii] * vanderY[:, jj]
            sVal = vanderScalerX[:, ii] * vanderScalerY[:, jj]
            ppp = get_upper_triangular_index(jj, ii)

            XModel[:, ppp] = pVal
            scaler[:, ppp] = sVal

    return XModel, scaler.flatten()


def estimate_mean_and_covariance_matrix_robust(x, w):
    """
    Estimate the weighted mean and covariance matrix of a dataset robustly.

    Parameters:
    - x (numpy.ndarray): Input data matrix, where each row represents a data point.
    - w (numpy.ndarray): Weight vector corresponding to the data points.

    Returns:
    - tuple: A tuple containing the weighted mean (mean) and the weighted covariance matrix (cov).

    Notes:
    - This function calculates the weighted mean and covariance matrix of a dataset.
    - Weighted statistics give more significance to data points with higher weights.
    - The weighted mean is calculated using the specified weights along each dimension.
    - The weighted covariance matrix is computed with the unbiased estimator (ddof=0) for sample covariance,
      adjusted for the sum of weights to make it more robust.

    Example:
    >>> import numpy as np
    >>> data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    >>> weights = np.array([0.5, 1.0, 0.5, 1.0])
    >>> estimated_mean, estimated_covariance = estimate_mean_and_covariance_matrix_robust(data, weights)
    """
    sum_w = np.sum(w)

    mean = np.average(x, weights=w, axis=0)
    cov = np.cov(x.T, ddof=0, aweights=w) * sum_w / (sum_w - 1)

    return mean, cov


def get_mahalanobis_distances(x, mean, invCov):
    """
    Calculate Mahalanobis distances of data points from a given mean using the inverse covariance matrix.

    Parameters:
    - x (numpy.ndarray): Data matrix where each row represents a data point.
    - mean (numpy.ndarray): The mean vector.
    - invCov (numpy.ndarray): The inverse of the covariance matrix.

    Returns:
    - numpy.ndarray: An array containing Mahalanobis distances for each data point.

    Notes:
    - Mahalanobis distance is a measure of the distance between a data point and a distribution, taking into account
      the correlation between variables and the variance of the distribution.
    - This function calculates Mahalanobis distances for each data point in the input matrix x, relative to the given mean
      and the inverse covariance matrix invCov.
    - Mahalanobis distances are used to detect outliers or evaluate how far data points are from the distribution center.

    Example:
    >>> import numpy as np
    >>> data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    >>> mean_vector = np.array([2.5, 3.5])
    >>> inv_covariance_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> distances = get_mahalanobis_distances(data, mean_vector, inv_covariance_matrix)
    """
    distances = np.zeros((x.shape[0],))
    for i in range(x.shape[0]):
        distances[i] = distance.mahalanobis(x[i], mean, invCov)

    return distances


def w_decay(z):
    """
    Returns the downweighting factor for normalized deviation z, using a modified exponential decay function. It effectively gives no weight to data points more than 10-20 sigmas from the fitted value (whereas w_huber gives significant weight at these distances).

    Parameters:
    - z (float or numpy.ndarray): Normalized residual(s) expected to be approximately N(0,1).

    Returns:
    - float or numpy.ndarray: Weight reduction factor(s) for the statistical weight.

    Example:
    >>> import numpy as np
    >>> normalized_residual = 2.5
    >>> weight_reduction_factor = w_decay(normalized_residual)
    """
    if isinstance(z, (list, np.ndarray)):
        weights = np.zeros(z.size)
        for i in range(weights.size):
            weights[i] = w_decay(z[i])
        return weights.flatten()
    else:
        z_abs = np.absolute(z)

        if (z_abs <= 2.0):
            return 1.0
        elif (z_abs <= 3.0):
            t = z_abs - 2.0
            return 1.0 - (1.77373519609519 - 1.14161463726663*t)*t*t;
        elif (z_abs <= 10.0):
            return np.exp(-z_abs/3.0)
        else:
            return 0.0