#!/usr/bin/env python
u"""
fit.py
Written by Tyler Sutterley (04/2024)

Fits a polynomial surface to a set of points

CALLING SEQUENCE:
    tsbeta = fit.poly_surface(t, x, y, data, ORDER_TIME=3, 
        ORDER_SPACE=3)
    reg_coef = tsbeta['beta']
    reg_error = tsbeta['error']

INPUTS:
    t_in: input time array
    x_in: x-coordinate array
    y_in: y-coordinate array
    d_in: input data array

OUTPUTS:
    beta: regressed coefficients array
    data: modeled elevation at centroid
    model: modeled surface time-series
    error: regression fit error for each coefficient for an input deviation
        STDEV: standard deviation of output error
        CONF: confidence interval of output error
    std_err: standard error for each coefficient
    R2: coefficient of determination (r**2).
        Proportion of variability accounted by the model
    R2Adj: adjusted r**2. adjusts the r**2 for the number of terms in the model
    MSE: mean square error
    WSSE: Weighted sum of squares error
    NRMSE: normalized root mean square error
    AIC: Akaike information criterion (Second-Order, AICc)
    BIC: Bayesian information criterion (Schwarz criterion)
    LOGLIK: log likelihood
    residual: model residual
    DOF: degrees of freedom
    N: number of terms used in fit
    cov_mat: covariance matrix
    centroid: centroid point of input coordinates

OPTIONS:
    RELATIVE: relative period
    FIT_TYPE: type of time-variable polynomial fit to apply
        ('polynomial', 'chebyshev', 'spline')
    ORDER_TIME: maximum polynomial order in time-variable fit
        (0=constant, 1=linear, 2=quadratic)
    ORDER_SPACE:  maximum polynomial order in spatial fit
        (1=planar, 2=quadratic)
    TERMS: list of extra terms
    STDEV: standard deviation of output error
    CONF: confidence interval of output error
    AICc: use second order AIC

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)
    scipy: Scientific Tools for Python (https://docs.scipy.org/doc/)

UPDATE HISTORY:
    Updated 04/2024: rewritten for python3 and added function docstrings
        add optional TERMS argument to augment the design matrix
        add spline design matrix option for time-variable fit
    Updated 09/2017: using rcond=-1 in numpy least-squares algorithms
        use median statistics for reducing to valid points
    Written 03/2014
"""
from __future__ import print_function, annotations

import numpy as np
import scipy.stats
from scipy.interpolate import BSpline

# PURPOSE: iteratively fit a polynomial surface to the elevation data to
# reduce to within a valid window
def reduce_fit(t_in, x_in, y_in, d_in, TERMS=[], **kwargs):
    """
    Iteratively fit a polynomial surface to the elevation data to reduce to
    within a valid surface window [Schenk2012]_ [Smith2019]_ [Sutterley2014]_

    Parameters
    ----------
    t_in: np.ndarray
        input time array
    x_in: np.ndarray
        x-coordinate array
    y_in: np.ndarray
        y-coordinate array
    d_in: np.ndarray
        input data array
    FIT_TYPE: str
        type of time-variable polynomial fit to apply

        - ``'polynomial'``
        - ``'chebyshev'``
        - ``'spline'``
    ORDER_TIME: int
        maximum polynomial order in time-variable fit
    ORDER_SPACE: int
        maximum polynomial order in spatial fit
    TERMS: list
        list of extra terms
    STDEV: float
        standard deviation of output error
    CONF: float
        confidence interval of output error
    AICc: bool
        use second order AIC
    ITERATE: int, default 25
        maximum number of iterations to use in fit
    kwargs: dict
        keyword arguments for the fit type

    References
    ----------
    .. [Schenk2012] T. Schenk and B. M. Csatho, "A New Methodology for Detecting
        Ice Sheet Surface Elevation Changes From Laser Altimetry Data",
        *IEEE Transactions on Geoscience and Remote Sensing*, 50(9),
        3302--3316, (2012). `doi:10.1109/TGRS.2011.2182357
        <https://doi.org/10.1109/TGRS.2011.2182357>`_
    .. [Smith2019] B. E. Smith el al., "Land ice height-retrieval
        algorithm for NASA's ICESat-2 photon-counting laser altimeter",
        *Remote Sensing of Environment*, 233, 111352, (2019).
        `doi:10.1016/j.rse.2019.111352
        <https://doi.org/10.1016/j.rse.2019.111352>`_
    .. [Sutterley2014] T. C. Sutterley, I. Velicogna, E. J. Rignot, J. Mouginot,
        T. Flament, M. R. van den Broeke, J. M. van Wessem, C. H. Reijmer, 
        "Mass loss of the Amundsen Sea Embayment of West Antarctica from
        four independent techniques", *Geophysical Research Letters*,
        41(23), 8421--8428, (2014). `doi:10.1002/2014GL061940 
        <https://doi.org/10.1002/2014GL061940>`_
    """
    kwargs.setdefault('ITERATE', 25)
    kwargs.setdefault('FIT_TYPE', 'polynomial')
    kwargs.setdefault('ORDER_TIME', 3)
    kwargs.setdefault('ORDER_SPACE', 3)
    # number of points for fit and number of terms in fit
    n_max = len(d_in)
    n_terms = (kwargs['ORDER_TIME'] + 1) + \
        np.sum(np.arange(2, kwargs['ORDER_SPACE'] + 2)) + \
        len(TERMS)
    # run only if number of points is above number of terms
    FLAG1 = ((n_max - n_terms) > 10)
    # set initial window to the full data range
    window = d_in.max() - d_in.min()
    window_p1 = np.copy(window)
    # initial indices for reducing to window
    filt = np.arange(n_max)
    filt_p1 = np.copy(filt)
    filt_p2 = -1*np.ones_like(filt)
    # indices of valid points
    ind = np.arange(n_max)
    if FLAG1:
        # save initial indices
        indices = ind.copy()
        # run fit program for fit types
        s = surface_fit(t_in, x_in, y_in, d_in, **kwargs)
        # number of iterations performed
        n_iter = 1
        # save beta coefficients
        beta_mat = np.copy(s['beta'])
        error_mat = np.copy(s['error'])
        data = np.copy(s['data'])
        model = np.copy(s['model'])
        # residuals of model fit
        resid = s['residual']
        # standard deviation of the residuals
        resid_std = np.std(resid)
        # save MSE and DOF for error analysis
        MSE = np.copy(s['MSE'])
        DOF = np.copy(s['DOF'])
        # Root mean square error
        RMSE = np.sqrt(s['MSE'])
        # Normalized root mean square error
        NRMSE = RMSE/(np.max(d_in)-np.min(d_in))
        # IQR pass: residual-(median value) is within 75% of IQR
        # RDE pass: residual-(median value) is within 50% of P84-P16
        IQR,RDE,MEDIAN = median_filter(resid)
        # checking if any residuals are outside of the window
        window = np.max([6.0*RDE, 0.5*window_p1])
        filt, = np.nonzero(np.abs(resid-MEDIAN) <= (window/2.0))
        # save iteration of window
        window_p1 = np.copy(window)
        # run only if number of points is above number of terms
        n_rem = np.count_nonzero(np.abs(resid-MEDIAN) <= (window/2.0))
        FLAG1 = ((n_rem - n_terms) > 10)
        # maximum number of iterations to prevent infinite loops
        FLAG2 = (n_iter <= kwargs['ITERATE'])
        # compare indices over two iterations to prevent false stoppages
        FLAG3 = (set(filt) != set(filt_p1)) | (set(filt_p1) != set(filt_p2)) 
        # iterate until there are no additional removed data points
        while FLAG1 & FLAG2 & FLAG3:
            # fit selected data for window
            t_filt = t_in[filt]
            x_filt = x_in[filt]
            y_filt = y_in[filt]
            d_filt = d_in[filt]
            indices = ind[filt]
            # reduce 
            terms = [t[filt] for t in TERMS]
            # run fit program for polynomial type
            s = surface_fit(t_filt, x_filt, y_filt, d_filt,
                TERMS=terms, **kwargs)
            # add to number of iterations performed
            n_iter += 1
            # save model coefficients
            beta_mat = np.copy(s['beta'])
            error_mat = np.copy(s['error'])
            data = np.copy(s['data'])
            model = np.copy(s['model'])
            # save MSE and DOF for error analysis
            MSE = np.copy(s['MSE'])
            DOF = np.copy(s['DOF'])
            # Root mean square error
            RMSE = np.sqrt(s['MSE'])
            # Normalized root mean square error
            NRMSE = RMSE/(np.max(d_filt)-np.min(d_filt))
            # save number of points
            n_max = len(d_filt)
            # residuals of model fit
            resid = s['residual']
            # standard deviation of the residuals
            resid_std = np.std(resid)
            # IQR pass: residual-(median value) is within 75% of IQR
            # RDE pass: residual-(median value) is within 50% of P84-P16
            IQR,RDE,MEDIAN = median_filter(resid)
            # checking if any residuals are outside of the window
            window = np.max([6.0*RDE, 0.5*window_p1])
            # filter out using median statistics and refit
            filt_p2 = np.copy(filt_p1)
            filt_p1 = np.copy(filt)
            filt, = np.nonzero(np.abs(resid-MEDIAN) <= (window/2.0))
            # save iteration of window
            window_p1 = np.copy(window)
            # run only if number of points is above number of terms
            n_rem = np.count_nonzero(np.abs(resid-MEDIAN) <= (window/2.0))
            FLAG1 = ((n_rem - n_terms) > 10)
            # maximum number of iterations to prevent infinite loops
            FLAG2 = (n_iter <= kwargs['ITERATE'])
            # compare indices over two iterations to prevent false stoppages
            FLAG3 = (set(filt) != set(filt_p1)) | (set(filt_p1) != set(filt_p2))

    # return reduced model fit
    FLAG3 = (set(filt) == set(filt_p1))
    if FLAG1 & FLAG3:
        return {'beta':beta_mat, 'error':error_mat, 'data':data,
            'model':model, 'MSE':MSE, 'NRMSE':NRMSE, 'DOF':DOF,
            'count':n_max, 'indices':indices, 'iterations':n_iter,
            'window':window, 'RDE':RDE, 'centroid':s['centroid']}
    else:
        raise ValueError(f'No valid data points found after {n_iter} iterations')

def surface_fit(t_in, x_in, y_in, d_in,
        FIT_TYPE='polynomial',
        ORDER_SPACE=3,
        TERMS=[],
        STDEV=0,
        CONF=0,
        AICc=True,
        **kwargs
    ):
    """
    Fits a polynomial surface to a set of points [Schenk2012]_ [Sutterley2014]_
    
    Parameters
    ----------
    t_in: np.ndarray
        input time array
    x_in: np.ndarray    
        x-coordinate array
    y_in: np.ndarray
        y-coordinate array
    d_in: np.ndarray
        input data array
    FIT_TYPE: str
        type of time-variable polynomial fit to apply

        - ``'polynomial'``
        - ``'chebyshev'``
        - ``'spline'``
    ORDER_TIME: int
        maximum polynomial order in time-variable fit
    ORDER_SPACE: int
        maximum polynomial order in spatial fit
    TERMS: list
        list of extra terms
    STDEV: float
        standard deviation of output error
    CONF: float
        confidence interval of output error
    AICc: bool
        use second order AIC
    kwargs: dict
        keyword arguments for the fit type

    References
    ----------
    .. [Schenk2012] T. Schenk and B. M. Csatho, "A New Methodology for Detecting
        Ice Sheet Surface Elevation Changes From Laser Altimetry Data",
        *IEEE Transactions on Geoscience and Remote Sensing*, 50(9),
        3302--3316, (2012). `doi:10.1109/TGRS.2011.2182357
        <https://doi.org/10.1109/TGRS.2011.2182357>`_
    .. [Sutterley2014] T. C. Sutterley, I. Velicogna, E. J. Rignot, J. Mouginot,
        T. Flament, M. R. van den Broeke, J. M. van Wessem, C. H. Reijmer, 
        "Mass loss of the Amundsen Sea Embayment of West Antarctica from
        four independent techniques", *Geophysical Research Letters*,
        41(23), 8421--8428, (2014). `doi:10.1002/2014GL061940 
        <https://doi.org/10.1002/2014GL061940>`_
    """

    # remove singleton dimensions from input variables
    t_in = np.squeeze(t_in)
    x_in = np.squeeze(x_in)
    y_in = np.squeeze(y_in)
    d_in = np.squeeze(d_in)
    nmax = len(t_in)
    # check that input dimensions match
    assert (len(x_in) == nmax) and (len(y_in) == nmax) and \
        (len(d_in) == nmax), 'Input dimensions do not match'

    # create design matrix for fit
    DMAT = []

    # time-variable design matrix
    if (FIT_TYPE.lower() == 'polynomial'):
        TMAT, t_rel = _polynomial(t_in, **kwargs)
    elif (FIT_TYPE.lower() == 'chebyshev'):
        TMAT = _chebyshev(t_in, **kwargs)
    elif (FIT_TYPE.lower() == 'spline'):
        TMAT = _spline(t_in, **kwargs)
    else:
        raise ValueError(f'Fit type {FIT_TYPE} not recognized')
    # append the time-variable design matrix
    DMAT.extend(TMAT)
    n_time = len(TMAT)

    # surface design matrix
    SMAT, centroid = _surface(x_in, y_in,
        ORDER_SPACE=ORDER_SPACE, **kwargs)
    DMAT.extend(SMAT)

    # add additional terms to the design matrix
    for t in TERMS:
        DMAT.append(t)
    # take the transpose of the design matrix
    DMAT = np.transpose(DMAT)

    # Standard Least-Squares fitting
    # (the [0] denotes coefficients output)
    beta_mat = np.linalg.lstsq(DMAT, d_in, rcond=-1)[0]
    n_terms = len(beta_mat)
    # Weights are equal
    wi = 1.0
    # modeled surface time-series
    mod = np.dot(DMAT, beta_mat)
    # modeled data at centroid
    data = np.dot(DMAT[:,:n_time], beta_mat[:n_time])
    # residual of fit
    res = d_in - np.dot(DMAT,beta_mat)

    # nu = Degrees of Freedom
    nu = nmax - n_terms

    # calculating R^2 values
    # SStotal = sum((Y-mean(Y))**2)
    SStotal = np.dot(np.transpose(d_in[0:nmax] - np.mean(d_in[0:nmax])),
        (d_in[0:nmax] - np.mean(d_in[0:nmax])))
    # SSerror = sum((Y-X*B)**2)
    SSerror = np.dot(np.transpose(d_in[0:nmax] - np.dot(DMAT,beta_mat)),
        (d_in[0:nmax] - np.dot(DMAT,beta_mat)))
    # R**2 term = 1- SSerror/SStotal
    rsquare = 1.0 - (SSerror/SStotal)
    # Adjusted R**2 term: weighted by degrees of freedom
    rsq_adj = 1.0 - (SSerror/SStotal)*np.float64((nmax-1.0)/nu)
    # Fit Criterion
    # number of parameters including the intercept and the variance
    K = np.float64(n_terms + 1)
    # Log-Likelihood with weights (if unweighted, weight portions == 0)
    # log(L) = -0.5*n*log(sigma^2) - 0.5*n*log(2*pi) - 0.5*n
    log_lik = 0.5*(np.sum(np.log(wi)) - nmax*(np.log(2.0 * np.pi) + 1.0 -
        np.log(nmax) + np.log(np.sum(wi * (res**2)))))

    # Aikaike's Information Criterion
    AIC = -2.0*log_lik + 2.0*K
    if AICc:
        # Second-Order AIC correcting for small sample sizes (restricted)
        # Burnham and Anderson (2002) advocate use of AICc where
        # ratio num/K is small
        # A small ratio is defined in the definition at approximately < 40
        AIC += (2.0*K*(K+1.0))/(nmax - K - 1.0)
    # Bayesian Information Criterion (Schwarz Criterion)
    BIC = -2.0*log_lik + np.log(nmax)*K

    # Regression with Errors with Unknown Standard Deviations
    # MSE = (1/nu)*sum((Y-X*B)**2)
    # Mean square error
    MSE = np.dot(np.transpose(d_in[0:nmax] - np.dot(DMAT,beta_mat)),
        (d_in[0:nmax] - np.dot(DMAT,beta_mat)))/np.float64(nu)
    # Root mean square error
    RMSE = np.sqrt(MSE)
    # Normalized root mean square error
    NRMSE = RMSE/(np.max(d_in[0:nmax]) - np.min(d_in[0:nmax]))
    # Covariance Matrix
    # Multiplying the design matrix by itself
    Hinv = np.linalg.inv(np.dot(np.transpose(DMAT), DMAT))
    # Taking the diagonal components of the covariance matrix
    hdiag = np.diag(Hinv)
    # set either the standard deviation or the confidence interval
    if (STDEV != 0):
        # Setting the standard deviation of the output error
        alpha = 1.0 - scipy.special.erf(STDEV/np.sqrt(2.0))
    elif (CONF != 0):
        # Setting the confidence interval of the output error
        alpha = 1.0 - CONF
    else:
        # Default is 95% confidence interval
        alpha = 1.0 - (0.95)
    # Student T-Distribution with D.O.F. nu
    # t.ppf parallels tinv in matlab
    tstar = scipy.stats.t.ppf(1.0-(alpha/2.0),nu)
    # beta_err is the error for each coefficient
    # beta_err = t(nu,1-alpha/2)*standard error
    st_err = np.sqrt(MSE*hdiag)
    beta_err = tstar*st_err

    # return the modeled surface time-series and the coefficients
    return {'beta':beta_mat, 'data':data, 'model':mod,
        'error':beta_err, 'std_err':st_err, 'R2':rsquare,
        'R2Adj':rsq_adj, 'MSE':MSE, 'NRMSE':NRMSE,
        'AIC':AIC, 'BIC':BIC, 'LOGLIK':log_lik,
        'residual':res, 'N':n_terms, 'DOF':nu,
        'cov_mat':Hinv, 'centroid':centroid}

# PURPOSE: calculate the interquartile range (Pritchard et al, 2009) and
# robust dispersion estimator (Smith et al, 2017) of the model residuals
def median_filter(r0):
    """
    Calculates the interquartile range [Pritchard2009]_ and
    robust dispersion estimator [Smith2017]_ of the model residuals

    Parameters
    ----------
    r0: float
        height residuals

    Returns
    -------
    IQR: float
        75% of the interquartile range
    RDE: float
        50% of the difference between the 84th and 16th percentiles
    median: float
        median value of height residuals

    References
    ----------
    .. [Pritchard2009] H. D. Pritchard et al., "Extensive dynamic thinning
        on the margins of the Greenland and Antarctic ice sheets",
        *Nature*, 461(7266), 971--975, (2009).
        `doi:10.1038/nature08471 <https://doi.org/10.1038/nature08471>`_
    .. [Smith2017] B. E. Smith el al., "Connected subglacial lake drainage
        beneath Thwaites Glacier, West Antarctica", *The Cryosphere*,
        11(1), 451--467, (2017).
        `doi:10.5194/tc-11-451-2017 <https://doi.org/10.5194/tc-11-451-2017>`_
    """
    # calculate percentiles for IQR and RDE
    # IQR: first and third quartiles (25th and 75th percentiles)
    # RDE: 16th and 84th percentiles
    # median: 50th percentile
    Q1,Q3,P16,P84,MEDIAN = np.percentile(r0,[25,75,16,84,50])
    # calculate interquartile range
    IQR = Q3 - Q1
    # calculate robust dispersion estimator (RDE)
    RDE = P84 - P16
    # IQR pass: residual-(median value) is within 75% of IQR
    # RDE pass: residual-(median value) is within 50% of P84-P16
    return (0.75*IQR, 0.5*RDE, MEDIAN)

def _polynomial(t_in, RELATIVE=Ellipsis, ORDER_TIME=3, **kwargs):
    """
    Create a polynomial design matrix for a time-series

    Parameters
    ----------
    t_in: np.ndarray
        input time array
    RELATIVE: int or np.ndarray
        relative period
    ORDER_TIME: int
        maximum polynomial order in time-variable fit

    Returns
    -------
    TMAT: list
        time-variable design matrix based on polynomial order
    t_rel: float
        relative time
    """
    # calculate epoch for calculating relative times
    if isinstance(RELATIVE, (list, np.ndarray)):
        t_rel = t_in[RELATIVE].mean()
    elif isinstance(RELATIVE, (float, int, np.float_, np.int_)):
        t_rel = np.copy(RELATIVE)
    elif (RELATIVE == Ellipsis):
        t_rel = t_in[RELATIVE].mean()
    # time-variable design matrix based on polynomial order
    TMAT = []
    # add polynomial orders (0=constant, 1=linear, 2=quadratic, etc)
    for o in range(ORDER_TIME+1):
        TMAT.append((t_in-t_rel)**o)
    # return the design matrix and the relative time
    return (TMAT, t_rel)

def _chebyshev(t_in, RELATIVE=None, ORDER_TIME=3, **kwargs):
    """
    Create a Chebyshev design matrix for a time-series
    
    Parameters
    ----------
    t_in: np.ndarray
        input time array
    RELATIVE: list or np.ndarray
        relative period
    ORDER_TIME: int
        maximum polynomial order in time-variable fit

    Returns
    -------
    TMAT: list
        time-variable design matrix based on polynomial order
    """
    # scale time-series to be [-1,1]
    # using either max and min of time-series or relative dates
    if RELATIVE is None:
        tmin = np.nanmin(t_in)
        tmax = np.nanmax(t_in)
        t_norm = ((t_in - tmin) - (tmax - t_in))/(tmax - tmin)
    elif isinstance(RELATIVE, (list, np.ndarray)):
        t_norm = ((t_in - RELATIVE[0]) - (RELATIVE[1]-t_in)) / \
            (RELATIVE[1] - RELATIVE[0])
    # time-variable design matrix based on polynomial order
    TMAT = []
    TMAT.append(np.ones_like(t_in))
    TMAT.append(t_norm)
    for o in range(2, ORDER_TIME+1):
        TMAT.append(2.0*t_norm*TMAT[o-1] - TMAT[o-2])
    # return the design matrix
    return TMAT

def _spline(t_in, KNOTS=[], ORDER_TIME=3, **kwargs):
    """
    Create a B-spline design matrix for a time-series
    
    Parameters
    ----------
    t_in: np.ndarray
        input time array
    KNOTS: list or np.ndarray
        Sorted 1D array of knots
    ORDER_TIME: int
        B-spline degree

    Returns
    -------
    TMAT: list
        time-variable design matrix based on polynomial order
    """
    # transpose the design matrix and add constant term
    TMAT = BSpline.design_matrix(t_in, KNOTS, ORDER_TIME,
        extrapolate=True).transpose().toarray()
    TMAT[0,:] = np.ones_like(t_in)
    # return the design matrix as a list
    return TMAT.tolist()

def _surface(x_in, y_in, ORDER_SPACE=3, **kwargs):
    """
    Create a surface design matrix for fit

    Parameters
    ----------
    x_in: np.ndarray
        x-coordinate array
    y_in: np.ndarray
        y-coordinate array
    ORDER_SPACE: int
        maximum polynomial order in spatial fit
    kwargs: dict
        keyword arguments for the fit type

    Returns
    -------
    SMAT: list
        surface design matrix
    centroid: dict
        centroid point of input coordinates
    """
    # calculate centroid point
    kwargs.setdefault('CX', np.mean(x_in))
    kwargs.setdefault('CY', np.mean(y_in))
    # calculate x and y relative to centroid point
    centroid = dict(x=kwargs['CX'], y=kwargs['CY'])
    rel_x = x_in - centroid['x']
    rel_y = y_in - centroid['y']
    # surface design matrix
    SMAT = []
    for o in range(1,ORDER_SPACE+1):
        for p in range(o+1):
            SMAT.append(rel_x**(o-p) * rel_y**p)
    # return the design matrix and the centroid
    return (SMAT, centroid)
