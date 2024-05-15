#!/usr/bin/env python
u"""
model_grounding_zone.py
Written by Tyler Sutterley (05/2024)
Creates a model of the tidal fluctuation of an ice shelf/sheet grounding zone
    using repeat period and error level of an input dataset (ATM/LVIS/ATL06)
Delineates the grounding zone using both a piecewise fit and an elastic model

Elastic model:
    D. G. Vaughan, Journal of Geophysical Research Solid Earth, 1995
        http://dx.doi.org/10.1029/94JB02467
    A. M. Smith, Journal of Glaciology, 1991
        https://doi.org/10.3198/1991JoG37-125-51-59
Piecewise fit:
    J. D. Toms and M. L. Lesperance, Ecology, 2003
        http://dx.doi.org/10.1890/02-0472

INPUTS:
    dataset: ATL06, ATM, LVIS

COMMAND LINE OPTIONS:
    -Y X, --year=X: Number of years to examine
    -D X, --dhdt=X: Proscibed elevation change rate
    -R, --reorient: Reorient elevation profile
    -V, --verbose: Verbose output of processing run

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: use argparse descriptions within documentation
    Updated 10/2021: use argparse to set command line options
    Updated 09/2017: use rcond=-1 in numpy least-squares algorithms
    Written 06/2017
"""
import sys
import os
import logging
import argparse
import numpy as np
import scipy.stats
import scipy.special
import scipy.optimize
import scipy.interpolate
import grounding_zones as gz

# attempt imports
plt = gz.utilities.import_dependency('matplotlib.pyplot')

# PURPOSE: test extracting grounding zone properties with different datasets
def model_grounding_zone(DATASET, n_years, dhdt, reorient, VERBOSE=False):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # create simulated flight line
    glacier_length = 150000
    x = np.linspace(0,glacier_length,1000).astype(np.float64)
    z = np.zeros_like(x)
    z[:475] = 240.0 - 2e-8*x[:475]**2
    z[474:525] = z[474] - 6.0*np.sin(np.arange(51)*np.pi/50)
    z[525:] = z[524]
    if reorient:
        z = z[::-1]

    # dataset parameters
    if DATASET in ('ATM','LVIS'):
        # repeat period
        repeat_period = 365
        spacing = 100.0
        # expected error level
        error_level = 0.20
        cutoff = 1.0*error_level
    elif (DATASET == 'ATL06'):
        # repeat period
        repeat_period = 91
        # expected error level
        error_level = 0.05
        spacing = 40.0
        cutoff = 2.0*error_level
    # number of measurements per year
    N = np.int64(365.0*n_years/repeat_period)

    # create interpolated flight line with smooth transitions
    SPL = scipy.interpolate.UnivariateSpline(x, z, k=4)
    XI = np.arange(0,glacier_length,spacing)
    NI = len(XI)
    ZI = SPL(XI)
    # # hydrostatic
    # FLOAT = np.int64(0.75*NI)
    # ZI[FLOAT:] = ZI[FLOAT]

    # constants for derivations of eta and beta parameters
    # D. G. Vaughan, Journal of Geophysical Research Solid Earth, 1995
    # A. M. Smith, Journal of Glaciology, 1991
    # density of water [kg/m^3]
    rho_w = 1030.0
    # density of ice [kg/m^3]
    rho_ice = 917.0
    # gravitational constant [m/s^2]
    g = 9.806
    # Poisson's ratio of ice
    nu = 0.3
    # Effective Elastic modulus of ice [Pa] (Table 1 of Vaughan)
    E = np.random.choice([9.,1.8,1.1,2.7,8.8,10.0,1.,2.,3.,0.83])*1e9
    # ice thickness of ice shelf [m]
    h = [1550.0,1500.0]

    # height iterations at time
    HI = np.zeros((NI,N+1))
    # mean
    MI = np.zeros((NI))
    # grounding points at each time
    gp = np.zeros((N+1), dtype=np.int64)
    # tidal amplitude at each time
    A = np.zeros((N+1), dtype=np.int64)
    # counter variable
    c = 0
    # amplitude of tidal signals (max = 2.4m from Padman, 2002)
    amp = 0.4 + 2.0#*np.random.sample()
    for t in range(0,n_years*365,repeat_period):
        # not taking into account specific phase of each constituent
        M2 = 1.0*np.sin(12.4206*t/24.0)*amp
        S2 = 0.46*np.sin(12.00*t/24.0)*amp
        O1 = 0.41*np.sin(12.00*t/24.0)*amp
        K1 = 0.40*np.sin(23.93*t/24.0)*amp
        N2 = 0.20*np.sin(12.66*t/24.0)*amp
        P1 = 0.19*np.sin(24.07*t/24.0)*amp
        L2 = 0.03*np.sin(12.19*t/24.0)*amp
        # composition of tides
        A[c] = M2 + S2 + O1 + K1 + N2 + P1 + L2
        # measurement uncertainty
        noise = error_level - 2.0*error_level*np.random.rand(NI)
        # add uncertainty to thinning between measurements
        dhdt_uncertainty = 0.0
        # dhdt_uncertainty = 0.05*thinning*np.random.rand(1)
        # calculation of elevation with thinning and noise
        HI[:,c] = np.copy(ZI) + t*(dhdt+dhdt_uncertainty)/365.25 + noise
        # add randomness to grounding line
        # gp[c] = np.int64(0.475*NI)
        gp[c], = np.random.randint(0.465*NI,0.485*NI,size=1)
        # # thickness profile of the ice shelf
        # H0 = np.linspace(h[0],h[1],len(X0))
        # H0[:,:] = h[0] - 0.0001*X0
        H0 = h[0]
        # distance from grounding line (0 = grounding line)
        d = (XI[:] - XI[gp[c]])
        R0 = np.zeros((NI))
        if np.sum(ZI[d < 0]) > np.sum(ZI[d > 0]):
            R0[d >= 0] = np.sqrt(d[d >= 0]**2)
        else:
            R0[d <= 0] = np.sqrt(d[d <= 0]**2)
        # structural rigidity of ice
        D = (E*H0**3)/(12.0*(1.0-nu**2))
        # beta elastic damping parameter
        b = (0.25*rho_w*g/D)**0.25
        # deflection of ice beyond the grounding line
        eta = A[c]*(1.0-np.exp(-b*R0)*(np.cos(b*R0) + np.sin(b*R0)))
        # create tidally oscillating surface elevation
        # grounding zone and hydrostatic ice shelf
        HI[:,c] += eta
        # add surface elevation at time t to mean
        MI += HI[:,c]
        # add 1 to counter
        c += 1

    # create plot
    fig, (ax1,ax2,ax3) = plt.subplots(num=1,nrows=3,sharex=True,figsize=(12,8))
    for i in range(N):
        # plot surface elevation at time t
        ll, = ax1.plot(XI,HI[:,i])
        # thickness profile of the ice shelf
        # H0 = np.linspace(h[0],h[1],len(XI[gp[i]:]))
        # ax1.plot(XI[gp[i]:], HI[gp[i]:,i]-H0, color=ll.get_color())
        # plot residual of surface elevation - mean
        ax2.plot(XI, HI[:,i]-MI/c, color=ll.get_color())
        # plot differentials
        CX = (XI[0:-1]+XI[1:])/2.0
        DYDX = (HI[1:,i]-HI[0:-1,i])/(XI[1:]-XI[0:-1])
        ax3.plot(CX, DYDX, color=ll.get_color())
        # if the tidal signal is large enough
        STDEV = np.std(HI[:,i]-MI/c)
        if (STDEV > cutoff):
            CONF = scipy.special.erf(2.0/np.sqrt(2.0))
            # fit with a hard piecewise model to get a rough estimate of GZ
            C1,C2,PWMODEL = piecewise_fit(XI, HI[:,i]-MI/c, STEP=100, CONF=CONF)
            ax2.plot(XI, PWMODEL, lw=0.5, color=ll.get_color()) if VERBOSE else None

            # distance from grounding line (0 = grounding line)
            d = (XI[:] - C1[0]).astype(np.int64)
            ORIENTATION=True if np.sum(ZI[d < 0]) < np.sum(ZI[d > 0]) else False
            # fit physical elastic model
            PGZ,PA,PE,PT,PdH,PEMODEL = physical_elastic_model(XI, HI[:,i]-MI/c,
                GZ=C1, ORIENTATION=ORIENTATION, CONF=0.95)
            logging.info('Real grounding line {0:f}'.format(XI[gp[i]]))
            logging.info('Estimated Grounding zone {0:f} {1:f}'.format(PGZ[0], PGZ[1]))
            logging.info('Real Tidal amplitude {0:f}'.format(PA[0]))
            logging.info('Estimated Tidal amplitude {0:f}'.format(PA[0]))
            logging.info('Ice shelf thickness {0:f}\n'.format(PT[0]))
            ax2.axvline(PGZ[0], color=ll.get_color(), ls='dashed', dashes=(11,5))
            ax2.axvspan(PGZ[0]-PGZ[1],PGZ[0]+PGZ[1],color=ll.get_color(),alpha=0.25)
            ax2.plot(XI, PEMODEL, color=ll.get_color())
            # plot actual modeled grounding line
            ax2.plot(XI[gp[i]], (HI[:,i]-MI/c)[gp[i]], '*', color=ll.get_color())
            # find inflexion point between grounding line and hydrostatic point
            jj, = np.nonzero((np.sign(DYDX[:-1]) < 0) & (np.sign(DYDX[1:]) >= 0) &
                (CX[:-1] >= C1[0]) & (CX[1:] <= C2[0]))
            ax3.axvline(np.mean(CX[jj]),color=ll.get_color(),ls='dashed',dashes=(11,5))

    # add mean to plot
    ax1.plot(XI, MI/c, 'k')

    # adjust subplots and show figure
    fig.subplots_adjust(left=0.05,right=0.98,bottom=0.04,top=0.98,hspace=0.05)
    plt.show()

# Derivation of Sharp Breakpoint Piecewise Regression:
# http://www.esajournals.org/doi/abs/10.1890/02-0472
# y = beta_0 + beta_1*t + e (for x <= alpha)
# y = beta_0 + beta_1*t + beta_2*(t-alpha) + e (for x > alpha)
def piecewise_fit(x, y, STEP=1, CONF=0.95):
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
    cutoff_array = np.zeros((n_param,2),dtype=np.int64)
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
            #- Least-Squares fitting (the [0] denotes coefficients output)
            beta_mat = np.linalg.lstsq(DMAT,YI,rcond=-1)[0]
            #- number of terms in least-squares solution
            n_terms = len(beta_mat)
            #- nu = Degrees of Freedom
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
    # CI1 = conf_interval(XI, PDF1/np.sum(PDF1), CONF)
    CI1 = 5000
    CMN1,CMX1 = (XI[n]-CI1,XI[nn]+CI1)
    # CI2 = conf_interval(XI, PDF2/np.sum(PDF2), CONF)
    CI2 = 5000
    CMN2,CMX2 = (XI[nn]-CI2,XI[nn]+CI2)

    # calculate model using best fit coefficients
    P_x0 = np.ones_like(x)
    P_x1a = np.copy(x)
    P_x1b = np.zeros_like(x)
    P_x1c = np.zeros_like(x)
    P_x1b[n*STEP:] = x[n*STEP:] - XI[n]
    P_x1c[nn*STEP:] = x[nn*STEP:] - XI[nn]
    DMAT = np.transpose([P_x0, P_x1a, P_x1b, P_x1c])
    beta_mat, = beta_matrix[ind,:]
    MODEL = np.dot(DMAT,beta_mat)

    # # create plots
    # gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    # ax1 = plt.subplot(gs[:,0])
    # ax2 = plt.subplot(gs[0,1])
    # ax3 = plt.subplot(gs[1,1], sharex=ax2)
    # im = ax1.imshow(loglik_matrix)
    # ax1.plot(nn,n,'*')
    # plt.colorbar(im, ax=ax1, orientation='horizontal')
    # ax2.plot(XI, PDF1/np.sum(PDF1))
    # ax2.vlines(XI[n], 0, np.max(PDF1), colors='red')
    # ax2.vlines(CMN1, 0, np.max(PDF1), colors='gray',linestyles='dashed')
    # ax2.vlines(CMX1, 0, np.max(PDF1), colors='gray',linestyles='dashed')
    # ax3.plot(XI, PDF2/np.sum(PDF1))
    # ax3.vlines(XI[nn], 0, np.max(PDF2), colors='red')
    # ax3.vlines(CMN2, 0, np.max(PDF2), colors='gray',linestyles='dashed')
    # ax3.vlines(CMX2, 0, np.max(PDF2), colors='gray',linestyles='dashed')
    # plt.show()
    # return the cutoff parameters, their confidence interval and the model
    return [XI[n],CMN1,CMX1], [XI[nn],CMN2,CMX2], MODEL

# PURPOSE: run a physical elastic bending model with Levenberg-Marquardt
# D. G. Vaughan, Journal of Geophysical Research Solid Earth, 1995
# A. M. Smith, Journal of Glaciology, 1991
# density of water [kg/m^3]
def physical_elastic_model(XI,YI,GZ=[0,0,0],METHOD='trf',ORIENTATION=False,CONF=0.95):
    # reorient input parameters to go from land ice to floating
    if ORIENTATION:
        Xm1 = XI[-1]
        GZ = Xm1 - GZ
        GZ[1:] = GZ[:0:-1]
        XI = Xm1 - XI[::-1]
        YI = YI[::-1]
    # elastic model parameters
    # G0: location of grounding line
    # A0: tidal amplitude (values from Padman 2002)
    # E0: Effective Elastic modulus of ice [Pa]
    # T0: ice thickness of ice shelf [m]
    # dH0: mean height change (thinning/thickening)
    p0 = [GZ[0], 1.2, 1e9, 1000.0, 0.0]
    # tuple for parameter bounds (lower and upper)
    # G0: 95% confidence interval of initial fit
    # A0: greater than +/- 2.4m value from Padman (2002)
    # E0: Range from Table 1 of Vaughan (1995)
    # T0: Range of ice thicknesses from Chuter (2015)
    # dH0: mean height change +/- 10 m/yr
    bounds = ([GZ[1], -3.0, 8.3e8, 100, -10], [GZ[2], 3.0, 1e10, 1900, 10])
    # optimized curve fit with Levenberg-Marquardt algorithm
    popt,pcov = scipy.optimize.curve_fit(elasticmodel, XI, YI,
        p0=p0, bounds=bounds, method=METHOD)
    MODEL = elasticmodel(XI, *popt)
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
    return (GZ,A,E,T,dH,MODEL)

# PURPOSE: create physical elastic bending model with a mean height change
def elasticmodel(x, GZ, A, E, T, dH):
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
    return dH + eta

# PURPOSE: calculate the confidence interval in the retrieval
def conf_interval(x,f,p):
    # sorting probability distribution from smallest probability to largest
    ii = np.argsort(f)
    # # compute the sorted cumulative probability distribution
    cdf = np.cumsum(f[ii])
    # find the min and max interval that contains the probability
    jj = np.max(np.nonzero(cdf < p))
    kk = np.min(np.nonzero(cdf >= p))
    # linearly interpolate to confidence interval
    J = x[ii[jj]] + (p - cdf[jj])/(cdf[kk] - cdf[jj])*(x[ii[kk]] - x[ii[jj]])
    K = x[ii[-1]]
    return np.abs(K-J)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates a model of the tidal fluctuation of an
            ice shelf/sheet grounding zone
            """
    )
    # command line parameters
    parser.add_argument('dataset',
        type=str, nargs='?',
        default='ATL06', choices=('ATL06','ATM','LVIS'),
        help='Input dataset')
    # number of years to examine
    parser.add_argument('--year','-Y',
        type=int, default=3,
        help='Number of years of measurements')
    # elevation change rate
    parser.add_argument('--dhdt','-D',
        type=float, default=2.0,
        help='Proscibed elevation change rate')
    # reorient elevation profile
    parser.add_argument('--reorient','-R',
        default=False, action='store_true',
        help='Reorient elevation profile')
    # verbosity settings
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of processing run')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # run model with parameters
    model_grounding_zone(args.dataset, args.year, args.dhdt,
        args.reorient, VERBOSE=args.verbose)

# run main program
if __name__ == '__main__':
    main()
