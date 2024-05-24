#!/usr/bin/env python
u"""
fit.py
Written by Tyler Sutterley (05/2024)

Utilities for creating models from surface elevation data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)
    scipy: Scientific Tools for Python (https://docs.scipy.org/doc/)

UPDATE HISTORY:
    Updated 05/2024: add function to build the complete design matrix
        add function to build the constraints for the least-squares fit
        add function to validate the columns in the design matrix
        add functions to give the number of spatial and temporal terms
        optionally use a bounded least-squares fit for the model runs
        add grounding zone, elastic bending and breakpoint fits
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
import scipy.optimize
from scipy.interpolate import BSpline

# PURPOSE: iteratively fit a polynomial surface to the elevation data to
# reduce to within a valid window
def iterative_surface(t_in, x_in, y_in, d_in, TERMS=[], **kwargs):
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
    ITERATIONS: int, default 25
        maximum number of iterations to use in fit
    ORDER_TIME: int
        maximum polynomial order in time-variable fit
    ORDER_SPACE: int
        maximum polynomial order in spatial fit
    TERMS: list
        list of extra terms
    BOUNDED: bool
        use bounded least-squares fit
    STDEV: float
        standard deviation of output error
    CONF: float
        confidence interval of output error
    AICc: bool
        use second order AIC
    kwargs: dict
        keyword arguments for the fit type

    Returns
    -------
    beta: np.ndarray
        regressed coefficients array
    error: np.ndarray
        regression fit error for each coefficient
    data: np.ndarray
        modeled elevation at centroid
    model: np.ndarray
        modeled surface time-series at input points
    std_error: np.ndarray
        standard error for each coefficient
    R2: float
        coefficient of determination (r^2)
    R2Adj: float
        adjusted r^2 value
    MSE: float
        mean square error
    WSSE: float
        weighted sum of squares error
    NRMSE: float
        normalized root mean square error
    AIC: float
        Akaike information criterion (Second-Order, AICc)
    BIC: float
        Bayesian information criterion (Schwarz criterion)
    LOGLIK: float
        log likelihood
    residual: np.ndarray
        model residual
    DOF: int
        degrees of freedom
    count: int
        final number of points used in the fit
    indices: np.ndarray
        indices of valid points
    iterations: int
        number of iterations performed
    window: float
        final window size for the fit
    RDE: float
        robust dispersion estimate
    centroid: dict
        centroid point used in the fit

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
    kwargs.setdefault('FIT_TYPE', 'polynomial')
    kwargs.setdefault('ITERATIONS', 25)
    kwargs.setdefault('MINIMUM_WINDOW', 1.0)
    kwargs.setdefault('MAXIMUM_RDE', 20.0)
    kwargs.setdefault('ORDER_TIME', 3)
    kwargs.setdefault('ORDER_SPACE', 3)
    kwargs.setdefault('KNOTS', [])
    kwargs.setdefault('THRESHOLD', 10)
    # number of points for fit
    n_max = len(d_in)
    # total number of spatial and temporal terms
    n_space = _spatial_terms(**kwargs)
    n_time = _temporal_terms(**kwargs)
    # total number of terms in fit
    n_terms = n_space + n_time + len(TERMS)
    # threshold for minimum number of points for fit
    # run only if number of points is above number of terms
    FLAG1 = ((n_max - n_terms) > kwargs['THRESHOLD'])
    # set initial window to the full data range
    window = d_in.max() - d_in.min()
    window_p1 = np.copy(window)
    h_min_win = np.copy(kwargs['MINIMUM_WINDOW'])
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
        s = polynomial_surface(t_in, x_in, y_in, d_in, **kwargs)
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
        # standard error
        std_error = np.copy(s['std_error'])
        # coefficients of determination
        rsquare = np.copy(s['R2'])
        rsq_adj = np.copy(s['R2Adj'])
        # save MSE and DOF for error analysis
        MSE = np.copy(s['MSE'])
        DOF = np.copy(s['DOF'])
        # Root mean square error
        RMSE = np.sqrt(s['MSE'])
        # Normalized root mean square error
        NRMSE = RMSE/(np.max(d_in)-np.min(d_in))
        # fit criterion
        AIC = np.copy(s['AIC'])
        BIC = np.copy(s['BIC'])
        log_lik = np.copy(s['LOGLIK'])
        # IQR pass: residual-(median value) is within 75% of IQR
        # RDE pass: residual-(median value) is within 50% of P84-P16
        IQR, RDE, MEDIAN = median_filter(resid)
        # checking if any residuals are outside of the window
        window = np.max([h_min_win, 6.0*RDE, 0.75*window_p1])
        filt, = np.nonzero(np.abs(resid - MEDIAN) <= (window/2.0))
        # save iteration of window
        window_p1 = np.copy(window)
        # run only if number of points is above number of terms
        n_rem = np.count_nonzero(np.abs(resid - MEDIAN) <= (window/2.0))
        FLAG1 = ((n_rem - n_terms) > kwargs['THRESHOLD'])
        # maximum number of iterations to prevent infinite loops
        FLAG2 = (n_iter <= kwargs['ITERATIONS'])
        # compare indices over two iterations to prevent false stoppages
        FLAG3 = (set(filt) != set(filt_p1)) | (set(filt_p1) != set(filt_p2))
        # compare robust dispersion estimate with maximum allowable
        FLAG4 = (RDE >= kwargs['MAXIMUM_RDE'])
        # iterate until there are no additional removed data points
        while FLAG1 & FLAG2 & (FLAG3 | FLAG4):
            # fit selected data for window
            t_filt = t_in[filt]
            x_filt = x_in[filt]
            y_filt = y_in[filt]
            d_filt = d_in[filt]
            indices = ind[filt]
            # reduce 
            terms = [t[filt] for t in TERMS]
            # run fit program for polynomial type
            s = polynomial_surface(t_filt, x_filt, y_filt, d_filt,
                TERMS=terms, **kwargs)
            # add to number of iterations performed
            n_iter += 1
            # save model coefficients
            beta_mat = np.copy(s['beta'])
            error_mat = np.copy(s['error'])
            data = np.copy(s['data'])
            model = np.copy(s['model'])
            # save number of points
            n_max = len(d_filt)
            # residuals of model fit
            resid = s['residual']
            # standard deviation of the residuals
            resid_std = np.std(resid)
            # standard error
            std_error = np.copy(s['std_error'])
            # coefficients of determination
            rsquare = np.copy(s['R2'])
            rsq_adj = np.copy(s['R2Adj'])
            # save MSE and DOF for error analysis
            MSE = np.copy(s['MSE'])
            DOF = np.copy(s['DOF'])
            # Root mean square error
            RMSE = np.sqrt(s['MSE'])
            # Normalized root mean square error
            NRMSE = RMSE/(np.max(d_filt)-np.min(d_filt))
            # fit criterion
            AIC = np.copy(s['AIC'])
            BIC = np.copy(s['BIC'])
            log_lik = np.copy(s['LOGLIK'])
            # IQR pass: residual-(median value) is within 75% of IQR
            # RDE pass: residual-(median value) is within 50% of P84-P16
            IQR, RDE, MEDIAN = median_filter(resid)
            # checking if any residuals are outside of the window
            window = np.max([h_min_win, 6.0*RDE, 0.75*window_p1])
            # filter out using median statistics and refit
            filt_p2 = np.copy(filt_p1)
            filt_p1 = np.copy(filt)
            filt, = np.nonzero(np.abs(resid - MEDIAN) <= (window/2.0))
            # save iteration of window
            window_p1 = np.copy(window)
            # run only if number of points is above number of terms
            n_rem = np.count_nonzero(np.abs(resid - MEDIAN) <= (window/2.0))
            FLAG1 = ((n_rem - n_terms) > kwargs['THRESHOLD'])
            # maximum number of iterations to prevent infinite loops
            FLAG2 = (n_iter <= kwargs['ITERATIONS'])
            # compare indices over two iterations to prevent false stoppages
            FLAG3 = (set(filt) != set(filt_p1)) | (set(filt_p1) != set(filt_p2))
            # compare robust dispersion estimate with maximum allowable
            FLAG4 = (RDE >= kwargs['MAXIMUM_RDE'])

    # return reduced model fit
    FLAG3 = (set(filt) == set(filt_p1))
    if FLAG1 & FLAG3 & np.logical_not(FLAG4):
        return {'beta':beta_mat, 'error':error_mat, 'data':data,
            'model':model, 'std_error':std_error, 'R2':rsquare,
            'R2Adj':rsq_adj, 'MSE':MSE, 'NRMSE':NRMSE, 
            'AIC':AIC, 'BIC':BIC, 'LOGLIK':log_lik,
            'residual':resid, 'DOF':DOF, 'count':n_max,
            'indices':indices, 'iterations':n_iter,
            'window':window, 'RDE':RDE, 'centroid':s['centroid']}
    else:
        raise Exception(f'No valid fit found after {n_iter} iterations')

def polynomial_surface(t_in, x_in, y_in, d_in,
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
    KNOTS: list or np.ndarray
        Sorted 1D array of knots for time-variable spline fit
    TERMS: list
        list of extra terms
    BOUNDED: bool
        use bounded least-squares fit
    STDEV: float
        standard deviation of output error
    CONF: float
        confidence interval of output error
    AICc: bool
        use second order AIC
    kwargs: dict
        keyword arguments for the fit type

    Returns
    -------
    beta: np.ndarray
        regressed coefficients array
    error: np.ndarray
        regression fit error for each coefficient
    data: np.ndarray
        modeled elevation at centroid
    model: np.ndarray
        modeled surface time-series at input points
    std_error: np.ndarray
        standard error for each coefficient
    R2: float
        coefficient of determination (r^2)
    R2Adj: float
        adjusted r^2 value
    MSE: float
        mean square error
    WSSE: float
        weighted sum of squares error
    NRMSE: float
        normalized root mean square error
    AIC: float
        Akaike information criterion (Second-Order, AICc)
    BIC: float
        Bayesian information criterion (Schwarz criterion)
    LOGLIK: float
        log likelihood
    residual: np.ndarray
        model residual
    N: int
        number of terms in the model
    DOF: int
        degrees of freedom
    cov_mat: np.ndarray
        covariance matrix
    centroid: dict
        centroid point used in the fit

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
    # set default keyword arguments
    kwargs.setdefault('FIT_TYPE', 'polynomial')
    kwargs.setdefault('ORDER_TIME', 3)
    kwargs.setdefault('ORDER_SPACE', 3)
    kwargs.setdefault('KNOTS', [])
    kwargs.setdefault('BOUNDED', True)

    # remove singleton dimensions from input variables
    t_in = np.squeeze(t_in)
    x_in = np.squeeze(x_in)
    y_in = np.squeeze(y_in)
    d_in = np.squeeze(d_in)
    # check that input dimensions match
    assert (len(x_in) == len(t_in)) and (len(y_in) == len(t_in)) and \
        (len(d_in) == len(t_in)), 'Input dimensions do not match'

    # create design matrix for fit
    M, centroid = _build_design_matrix(t_in, x_in, y_in, **kwargs)
    # validate the design matrix
    DMAT, indices = _validate_design_matrix(M)
    # total number of temporal terms
    n_time = _temporal_terms(**kwargs)
    # total number of terms
    n_max, n_total = M.shape
    n_max, n_terms = DMAT.shape
    # nu = Degrees of Freedom
    nu = n_max - n_terms

    # build the constraints for the fit
    if kwargs['BOUNDED']:
        bounds = _build_constraints(t_in, x_in, y_in, d_in,
            INDICES=indices, **kwargs)
        max_iter = None
    else:
        bounds = (-np.inf, np.inf)
        max_iter = 1
    # use linear least-squares (with bounds on the variables)
    results = scipy.optimize.lsq_linear(DMAT, d_in,
        bounds=bounds, max_iter=max_iter)
    beta_mat = np.zeros((n_total))
    beta_mat[indices] = np.copy(results['x'])
    # estimated mean square error
    MSE = np.sum(results['fun']**2)/np.float64(nu)

    # Weights are equal
    wi = 1.0
    # modeled surface time-series
    mod = np.dot(DMAT, results['x'])
    # modeled data at centroid
    data = np.dot(M[:,:n_time], beta_mat[:n_time])
    # residual of fit
    res = d_in - np.dot(DMAT, results['x'])

    # calculating R^2 values
    # SStotal = sum((Y-mean(Y))**2)
    SStotal = np.dot(np.transpose(d_in[0:n_max] - np.mean(d_in[0:n_max])),
        (d_in[0:n_max] - np.mean(d_in[0:n_max])))
    # SSerror = sum((Y-X*B)**2)
    SSerror = np.dot(np.transpose(d_in[0:n_max] - np.dot(DMAT,results['x'])),
        (d_in[0:n_max] - np.dot(DMAT,results['x'])))
    # R**2 term = 1- SSerror/SStotal
    rsquare = 1.0 - (SSerror/SStotal)
    # Adjusted R**2 term: weighted by degrees of freedom
    rsq_adj = 1.0 - (SSerror/SStotal)*np.float64((n_max-1.0)/nu)
    # Fit Criterion
    # number of parameters including the intercept and the variance
    K = np.float64(n_terms + 1)
    # Log-Likelihood with weights (if unweighted, weight portions == 0)
    # log(L) = -0.5*n*log(sigma^2) - 0.5*n*log(2*pi) - 0.5*n
    log_lik = 0.5*(np.sum(np.log(wi)) - n_max*(np.log(2.0 * np.pi) + 1.0 -
        np.log(n_max) + np.log(np.sum(wi * (res**2)))))

    # Aikaike's Information Criterion
    AIC = -2.0*log_lik + 2.0*K
    if AICc:
        # Second-Order AIC correcting for small sample sizes (restricted)
        # Burnham and Anderson (2002) advocate use of AICc where
        # ratio num/K is small
        # A small ratio is defined in the definition at approximately < 40
        AIC += (2.0*K*(K+1.0))/(n_max - K - 1.0)
    # Bayesian Information Criterion (Schwarz Criterion)
    BIC = -2.0*log_lik + np.log(n_max)*K

    # Root mean square error
    RMSE = np.sqrt(MSE)
    # Normalized root mean square error
    NRMSE = RMSE/(np.max(d_in[0:n_max]) - np.min(d_in[0:n_max]))
    # Covariance Matrix
    # Multiplying the design matrix by itself
    Hinv = np.linalg.inv(np.dot(np.transpose(DMAT), DMAT))
    # Taking the diagonal components of the covariance matrix
    hdiag = np.zeros((n_total))
    hdiag[indices] = np.diag(Hinv)
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
    std_error = np.sqrt(MSE*hdiag)
    beta_err = tstar*std_error

    # return the modeled surface time-series and the coefficients
    return {'beta':beta_mat, 'error':beta_err, 'data':data,
        'model':mod, 'std_error':std_error, 'R2':rsquare,
        'R2Adj':rsq_adj, 'MSE':MSE, 'NRMSE':NRMSE,
        'AIC':AIC, 'BIC':BIC, 'LOGLIK':log_lik,
        'residual':res, 'N':n_terms, 'DOF':nu,
        'cov_mat':Hinv, 'centroid':centroid}

# Derivation of Sharp Breakpoint Piecewise Regression:
# http://www.esajournals.org/doi/abs/10.1890/02-0472
# y = beta_0 + beta_1*t + e (for x <= alpha)
# y = beta_0 + beta_1*t + beta_2*(t-alpha) + e (for x > alpha)
def piecewise_bending(x, y, STEP=1, CONF=None):
    """
    Fits a piecewise linear regression to elevation data
    to find two sharp breakpoints

    Parameters
    ----------
    x: np.ndarray
        input x-coordinate array
    y: np.ndarray
        input y-coordinate array
    STEP: int, default 1
        step size for regridding the input data
    CONF: float or None, default None
        confidence interval of output error

    Returns
    -------
    point1: list
        first breakpoint and confidence interval
    point2: list
        second breakpoint and confidence interval
    model: np.ndarray
        modeled surface from the piecewise fit
    """
    # regrid x and y to STEP
    XI = x[::STEP]
    YI = y[::STEP]
    # Creating Design matrix based on chosen input fit_type parameters:
    nmax = len(XI)
    P_x0 = np.ones((nmax))# Constant Term
    P_x1a = XI[0:nmax]# Linear Term 1
    # Calculating the number parameters to search
    n_param = (nmax**2 - nmax)//2
    # R^2 and Log-Likelihood
    rsquare_array = np.zeros((n_param))
    loglik_array = np.zeros((n_param))
    # output cutoff and fit parameters
    cutoff_array = np.zeros((n_param,2),dtype=int)
    beta_matrix = np.zeros((n_param,4))
    # counter variable
    c = 0
    # SStotal = sum((Y-mean(Y))^2)
    SStotal = np.dot(np.transpose(YI - np.mean(YI)),(YI - np.mean(YI)))
    # uniform distribution over entire range
    for n in range(0,nmax):
        # Linear Term 2 (= change from linear term1: trend2 = beta1+beta2)
        P_x1b = np.zeros((nmax))
        P_x1b[n:nmax] = XI[n:nmax] - XI[n]
        for nn in range(n+1,nmax):
            # Linear Term 3 (= change from linear term2)
            P_x1c = np.zeros((nmax))
            P_x1c[nn:nmax] = XI[nn:nmax] - XI[nn]
            DMAT = np.transpose([P_x0, P_x1a, P_x1b, P_x1c])
            # Calculating Least-Squares Coefficients
            # Least-Squares fitting (the [0] denotes coefficients output)
            beta_mat = np.linalg.lstsq(DMAT,YI,rcond=-1)[0]
            # number of terms in least-squares solution
            n_terms = len(beta_mat)
            # nu = Degrees of Freedom
            # number of measurements-number of parameters
            nu = nmax - n_terms
            # residual of data-model
            residual = YI - np.dot(DMAT,beta_mat)
            # CALCULATING R_SQUARE VALUES
            # SSerror = sum((Y-X*B)^2)
            SSerror = np.dot(np.transpose(residual),residual)
            # R^2 term = 1- SSerror/SStotal
            rsquare_array[c] = 1 - (SSerror/SStotal)
            # Log-Likelihood
            loglik_array[c] = 0.5*(-nmax*(np.log(2.0 * np.pi) + 1.0 - \
                np.log(nmax) + np.log(np.sum(residual**2))))
            # save cutoffs and beta matrix
            cutoff_array[c,:] = [n,nn]
            beta_matrix[c,:] = beta_mat
            # add 1 to counter
            c += 1

    # find where Log-Likelihood is maximum
    ind, = np.nonzero(loglik_array == loglik_array.max())
    n,nn = cutoff_array[ind,:][0]
    # create matrix of likelihoods
    likelihood = np.zeros((nmax,nmax))
    likelihood[:,:] = np.nan
    likelihood[cutoff_array[:,0],cutoff_array[:,1]] = np.exp(loglik_array) / \
        np.sum(np.exp(loglik_array))
    # probability distribution functions of each cutoff
    PDF1 = np.zeros((nmax))
    PDF2 = np.zeros((nmax))
    for i in range(nmax):
        # PDF for cutoff 1 for all cutoff 2
        PDF1[i] = np.nansum(likelihood[i,:])
        # PDF for cutoff 2 for all cutoff 1
        PDF2[i] = np.nansum(likelihood[:,i])
    # calculate confidence intervals
    if CONF is not None:
        CI1 = _confidence_interval(XI, PDF1/np.sum(PDF1), CONF)
        CI2 = _confidence_interval(XI, PDF2/np.sum(PDF2), CONF)
    else:
        # use a default confidence interval
        CI1 = 5e3
        CI2 = 5e3
    # confidence interval for cutoffs
    CMN1,CMX1 = (XI[n]-CI1, XI[n]+CI1)
    CMN2,CMX2 = (XI[nn]-CI2, XI[nn]+CI2)

    # calculate model using best fit coefficients
    P_x0 = np.ones_like(x)
    P_x1a = np.copy(x)
    P_x1b = np.zeros_like(x)
    P_x1c = np.zeros_like(x)
    P_x1b[n*STEP:] = x[n*STEP:] - XI[n]
    P_x1c[nn*STEP:] = x[nn*STEP:] - XI[nn]
    DMAT = np.transpose([P_x0, P_x1a, P_x1b, P_x1c])
    beta_mat, = beta_matrix[ind,:]
    MODEL = np.dot(DMAT, beta_mat)
    # return the cutoff points, their confidence interval and the model
    point1 = [XI[n], CMN1, CMX1]
    point2 = [XI[nn], CMN2, CMX2]
    return (point1, point2, MODEL)

# PURPOSE: run a physical elastic bending model with Levenberg-Marquardt
# D. G. Vaughan, Journal of Geophysical Research Solid Earth, 1995
# A. M. Smith, Journal of Glaciology, 1991
def elastic_bending(XI, YI,
        METHOD='trf',
        GRZ=[0,0,0],
        TIDE=[0,0,0],
        ORIENTATION=False,
        THICKNESS=None,
        CONF=0.95,
        XOUT=None
    ):
    """
    Fits an elastic bending model to the grounding zone of an
    ice shelf [Smith1991]_ [Vaughan1995]_

    Parameters
    ----------
    XI: np.ndarray
        input x-coordinate array
    YI: np.ndarray
        input y-coordinate array
    METHOD: str
        optimization algorithm to use in curve_fit
    GRZ: np.ndarray
        initial guess for the grounding line location
    TIDE: np.ndarray
        initial guess for the tidal amplitude
    ORIENTATION: bool
        reorient input parameters to go from land ice to floating
    THICKNESS: np.ndarray or None
        initial guess for ice thickness
    CONF: float, default 0.95
        confidence interval of output error
    XOUT: np.ndarray or None
        output x-coordinates for model fit

    Returns
    -------
    GZ: np.ndarray
        grounding line location and confidence interval
    A: np.ndarray
        tidal amplitude and confidence interval
    E: np.ndarray
        effective elastic modulus of ice and confidence interval
    T: np.ndarray
        ice thickness of ice shelf and confidence interval
    dH: np.ndarray
        mean height change and confidence interval
    MODEL: np.ndarray
        modeled surface from the elastic bending model

    References
    ----------
    .. [Smith1991] A. M. Smith, "The use of tiltmeters to study the
        dynamics of Antarctic ice-shelf grounding lines",
        *Journal of Glaciology*, 37(125), 51--58, (1991).
        `doi:10.3198/1991JoG37-125-51-59
        <https://doi.org/10.3198/1991JoG37-125-51-59>`_
    .. [Vaughan1995] D. G. Vaughan, "Tidal flexure at ice shelf margins",
        *Journal of Geophysical Research Solid Earth*, 100(B4),
        6213--6224, (1995). `doi:10.1029/94JB02467
        <https://doi.org/10.1029/94JB02467>`_
    """

    # default output x-coordinates for model fit
    if XOUT is None:
        XOUT = np.copy(XI)

    # reorient input parameters to go from land ice to floating
    if ORIENTATION:
        Xm1 = XI[-1]
        GRZ = Xm1 - GRZ
        GRZ[1:] = GRZ[:0:-1]
        XI = Xm1 - XI[::-1]
        YI = YI[::-1]
        XOUT = Xm1 - XOUT[::-1]

    # calculate thickness mean, min and max
    if THICKNESS is not None:
        # only use positive thickness values
        # ocean points could be negative with tides
        ii, = np.nonzero(THICKNESS > 0.0)
        MTH = np.mean(THICKNESS[ii])
        MNTH = np.min(THICKNESS[ii])
        MXTH = np.max(THICKNESS[ii])
    else:
        MTH = 1000.0
        MNTH = 100.0
        MXTH = 1900.0

    # elastic model parameters
    # G0: location of grounding line
    # A0: tidal amplitude (values from Padman 2002)
    # E0: Effective Elastic modulus of ice [Pa]
    # T0: ice thickness of ice shelf [m]
    # dH0: mean height change (thinning/thickening)
    p0 = [GRZ[0], TIDE[0], 1e9, MTH, 0.0]
    # tuple for parameter bounds (lower and upper)
    # G0: 95% confidence interval of initial fit
    # A0: greater than +/- 2.4m value from Padman (2002)
    # E0: Range from Table 1 of Vaughan (1995)
    # T0: Range of ice thicknesses from Chuter (2015)
    # dH0: mean height change +/- 10 m/yr
    bounds = ([GRZ[1], TIDE[1], 8.3e8, MNTH, -10],
        [GRZ[2], TIDE[2], 1e10, MXTH, 10])
    # optimized curve fit with Levenberg-Marquardt algorithm
    popt,pcov = scipy.optimize.curve_fit(_elastic, XI, YI,
        p0=p0, bounds=bounds, method=METHOD)
    MODEL = _elastic(XOUT, *popt)
    # elasticmodel function outputs and 1 standard deviation uncertainties
    GZ = np.zeros((2))
    A = np.zeros((2))
    E = np.zeros((2))
    T = np.zeros((2))
    dH = np.zeros((2))
    GZ[0],A[0],E[0],T[0],dH[0] = popt[:]
    # Error analysis
    # nu = Degrees of Freedom = number of measurements-number of parameters
    nu = len(XI) - len(p0)
    # Setting the confidence interval of the output error
    alpha = 1.0 - CONF
    # Student T-Distribution with D.O.F. nu
    # t.ppf parallels tinv in matlab
    tstar = scipy.stats.t.ppf(1.0-(alpha/2.0),nu)
    # error for each coefficient = t(nu,1-alpha/2)*standard error
    perr = np.sqrt(np.diag(pcov))
    GZ[1],A[1],E[1],T[1],dH[1] = tstar*perr[:]
    # reverse the reorientation
    if ORIENTATION:
        GZ[0] = Xm1 - GZ[0]
        MODEL = MODEL[::-1]
    # return the model outputs
    return (GZ, A, E, T, dH, MODEL)

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

def _build_design_matrix(t_in, x_in, y_in,
        FIT_TYPE='polynomial',
        ORDER_SPACE=3,
        TERMS=[],
        **kwargs,
    ):
    """
    Builds the complete design matrix for the surface fit

    Parameters
    ----------
    t_in: np.ndarray
        input time array
    x_in: np.ndarray    
        x-coordinate array
    y_in: np.ndarray
        y-coordinate array
    FIT_TYPE: str
        type of time-variable polynomial fit to apply

        - ``'polynomial'``
        - ``'chebyshev'``
        - ``'spline'``
    ORDER_TIME: int
        maximum polynomial order in time-variable fit
    ORDER_SPACE: int
        maximum polynomial order in spatial fit
    KNOTS: list or np.ndarray
        Sorted 1D array of knots for time-variable spline fit
    TERMS: list
        list of extra terms
    kwargs: dict
        keyword arguments for the fit type

    Returns
    -------
    DMAT: np.ndarray
        Design matrix for the fit type
    centroid: dict
        centroid point of input coordinates
    """
    # output design matrix
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
    # surface design matrix
    SMAT, centroid = _surface(x_in, y_in,
        ORDER_SPACE=ORDER_SPACE, **kwargs)
    DMAT.extend(SMAT)
    # add additional terms to the design matrix
    for t in TERMS:
        DMAT.append(t)
    # return the transpose of the design matrix and the centroid
    return np.transpose(DMAT), centroid

def _validate_design_matrix(DMAT):
    """
    Validates the design matrix for the surface fit

    Parameters
    ----------
    DMAT: np.ndarray
        Design matrix for the fit type

    Returns
    -------
    DMAT: np.ndarray
        Design matrix for the fit type
    indices: np.ndarray
        indices of valid columns in the design matrix
    """
    # indices of valid columns in the design matrix
    indices, = np.nonzero(np.any(DMAT != 0, axis=0))
    # return the design matrix and the indices
    return DMAT[:,indices], indices

def _build_constraints(t_in, x_in, y_in, d_in, **kwargs):
    """
    Builds the constraints for the surface fit

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
    KNOTS: list or np.ndarray
        Sorted 1D array of knots for time-variable spline fit
    TERMS: list
        list of extra terms
    INDICES: np.ndarray
        indices of valid columns in the design matrix
    kwargs: dict
        keyword arguments for the fit type

    Returns
    -------
    lb: np.ndarray
        Lower bounds for the fit
    ub: dict
        Upper bounds for the fit
    """
    # default keyword arguments
    kwargs.setdefault('INDICES', Ellipsis)
    kwargs.setdefault('TERMS', [])
    # indices of valid columns in the design matrix
    indices = kwargs['INDICES'].copy()
    # total number of spatial and temporal terms
    n_space = _spatial_terms(**kwargs)
    n_time = _temporal_terms(**kwargs)
    # total number of terms in fit
    n_terms = n_space + n_time + len(kwargs['TERMS'])
    # parameter bounds
    lb = np.full((n_terms), -np.inf)
    ub = np.full((n_terms), np.inf)
    # minimum and maximum values for data and time
    dmin = np.min(d_in)
    dmax = np.max(d_in)
    dsigma = np.std(d_in)
    tmin = np.min(t_in)
    tmax = np.max(t_in)
    # bounds for surface
    lb[0] = dmin - dsigma
    ub[0] = dmax + dsigma
    # time-variable constraints
    FIT_TYPE = kwargs['FIT_TYPE'].lower()
    if (FIT_TYPE == 'polynomial') and (n_time > 1):
        lb[1] = (dmin - dmax - 2.0*dsigma)/(tmax - tmin)
        ub[1] = (dmax - dmin + 2.0*dsigma)/(tmax - tmin)
    elif (FIT_TYPE == 'chebyshev'):
        pass
    elif (FIT_TYPE == 'spline'):
        # bounds for spline fit
        for i in range(1, n_time):
            lb[i] = dmin - dsigma
            ub[i] = dmax + dsigma
    else:
        raise ValueError(f'Fit type {FIT_TYPE} not recognized')
    # return the constraints
    return (lb[indices], ub[indices])

def _temporal_terms(**kwargs):
    """
    Calculates the number of temporal terms for a given fit

    Parameters
    ----------
    FIT_TYPE: str
        type of time-variable polynomial fit to apply

        - ``'polynomial'``
        - ``'chebyshev'``
        - ``'spline'``
    ORDER_TIME: int
        maximum polynomial order in time-variable fit
    KNOTS: list or np.ndarray
        Sorted 1D array of knots for time-variable spline fit

    Returns
    -------
    n_time: int
        Number of time-variable terms in fit
    """
    # calculate the number of temporal terms for a given fit
    if kwargs['FIT_TYPE'] in ('spline', ):
        n_time = len(kwargs['KNOTS']) - 2
    else:
        n_time = (kwargs['ORDER_TIME'] + 1)
    # return the number of temporal terms for fit
    return n_time

def _spatial_terms(**kwargs):
    """
    Calculates the number of spatial terms for a given fit

    Parameters
    ----------
    ORDER_SPACE: int
        maximum polynomial order in spatial fit

    Returns
    -------
    n_space: int
        Number of spatial terms in fit
    """
    n_space = np.sum(np.arange(2, kwargs['ORDER_SPACE'] + 2)) 
    # return the number of temporal terms for fit
    return n_space

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
        t_rel = np.mean(RELATIVE)
    elif isinstance(RELATIVE, (float, int, np.float_, np.int_)):
        t_rel = np.copy(RELATIVE)
    elif RELATIVE in (Ellipsis, None):
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

# PURPOSE: create physical elastic bending model with a mean height change
def _elastic(x, GZ, A, E, T, dH):
    """
    Physical elastic bending model with a mean height change

    Parameters
    ----------
    x: np.ndarray
        x-coordinate array
    GZ: float
        grounding line location
    A: float
        tidal amplitude
    E: float
        effective elastic modulus of ice
    T: float
        ice thickness of ice shelf
    dH: float
        mean height change

    Returns
    -------
    model: np.ndarray
        modeled surface from the elastic bending model
    """
    # density of water [kg/m^3]
    rho_w = 1030.0
    # gravitational constant [m/s^2]
    g = 9.806
    # Poisson's ratio of ice
    nu = 0.3
    # structural rigidity of ice
    D = (E*T**3)/(12.0*(1.0-nu**2))
    # beta elastic damping parameter
    b = (0.25*rho_w*g/D)**0.25
    # distance of points from grounding line (R0 = 0 at grounding line)
    R0 = (x[x >= GZ] - GZ)
    # deflection of ice beyond the grounding line (elastic)
    eta = np.zeros_like(x)
    eta[x >= GZ] = A*(1.0-np.exp(-b*R0)*(np.cos(b*R0) + np.sin(b*R0)))
    # model = large scale height change + tidal deflection
    model = (dH + eta)
    return model

# PURPOSE: calculate the confidence interval in the retrieval
def _confidence_interval(x, f, p):
    """
    Calculate the confidence interval

    Parameters
    ----------
    x: np.ndarray
        input x-coordinate array
    f: np.ndarray
        input probability distribution
    p: float
        confidence interval
    """
    # sorting probability distribution from smallest probability to largest
    ii = np.argsort(f)
    # compute the sorted cumulative probability distribution
    cdf = np.cumsum(f[ii])
    # linearly interpolate to confidence interval
    J = np.interp(p, cdf, x[ii])
    # position with maximum probability
    K = x[ii[-1]]
    return np.abs(K-J)
