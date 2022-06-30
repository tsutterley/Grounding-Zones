#!/usr/bin/env python
u"""
calculate_GZ_ICESat2_ATL11.py
Written by Tyler Sutterley (05/2022)
Calculates ice sheet grounding zones with ICESat-2 data following:
    Brunt et al., Annals of Glaciology, 51(55), 2010
        https://doi.org/10.3189/172756410791392790
    Fricker et al. Geophysical Research Letters, 33(15), 2006
        https://doi.org/10.1029/2006GL026907
    Fricker et al. Antarctic Science, 21(5), 2009
        https://doi.org/10.1017/S095410200999023X
Outputs an HDF5 file of flexure scaled to match the downstream tide model

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -T X, --tide X: Tide model to use in correction
        CATS0201
        CATS2008
        TPXO9-atlas
        TPXO9-atlas-v2
        TPXO9-atlas-v3
        TPXO9-atlas-v4
        TPXO9.1
        TPXO8-atlas
        TPXO7.2
        AODTM-5
        AOTIM-5
        AOTIM-5-2018
        GOT4.7
        GOT4.8
        GOT4.10
        FES2014
    -R X, --reanalysis X: Reanalysis model to run
        ERA-Interim: http://apps.ecmwf.int/datasets/data/interim-full-moda
        ERA5: http://apps.ecmwf.int/data-catalogues/era5/?class=ea
        MERRA-2: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
    -C, --crossovers: Run ATL11 Crossovers
    -P, --plot: Create plots of flexural zone
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python (Spatial algorithms and data structures)
        https://docs.scipy.org/doc/
        https://docs.scipy.org/doc/scipy/reference/spatial.html
    matplotlib: Python 2D plotting library
        http://matplotlib.org/
        https://github.com/matplotlib/matplotlib
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    fiona: Python wrapper for vector data access functions from the OGR library
        https://fiona.readthedocs.io/en/latest/manual.html
    shapely: PostGIS-ish operations outside a database context for Python
        http://toblerity.org/shapely/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    read_ICESat2_ATL11.py: reads ICESat-2 annual land ice height data files
    time.py: utilities for calculating time operations
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 05/2022: use argparse descriptions within documentation
        use tide model class to get available models and references
        output estimated elastic modulus in grounding zone data group
    Updated 03/2021: output HDF5 file of flexure scaled by a tide model
        estimate flexure for crossovers using along-track model outputs
        final extent of the flexure AT is the estimated grounding line
        output grounding zone data group to output fit statistics
        replaced numpy bool/int to prevent deprecation warnings
        use utilities to set default path to shapefiles
    Updated 01/2021: using standalone ATL11 reader
        using argparse to set command line options
        using time module for conversion operations
    Written 12/2020
"""
from __future__ import print_function

import sys
import os
import re
import h5py
import fiona
import pyproj
import datetime
import argparse
import operator
import itertools
import numpy as np
import collections
import scipy.stats
import scipy.optimize
import shapely.geometry
import matplotlib.pyplot as plt
import pyTMD.model
import icesat2_toolkit.time
from grounding_zones.utilities import get_data_path
from icesat2_toolkit.read_ICESat2_ATL11 import read_HDF5_ATL11, \
    read_HDF5_ATL11_pair

# grounded ice shapefiles
grounded_shapefile = {}
grounded_shapefile['N'] = 'grn_ice_sheet_peripheral_glaciers.shp'
grounded_shapefile['S'] = 'ant_ice_sheet_islands_v2.shp'
# description and reference for each grounded ice file
grounded_description = {}
grounded_description['N'] = 'Greenland Mapping Project (GIMP) Ice & Ocean Mask'
grounded_description['S'] = ('MEaSUREs Antarctic Boundaries for IPY 2007-2009 '
    'from Satellite_Radar, Version 2')
grounded_reference = {}
grounded_reference['N'] = 'https://doi.org/10.5194/tc-8-1509-2014'
grounded_reference['S'] = 'https://doi.org/10.5067/IKBWW4RYHF1Q'

# PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        projection_flag = 'S'
    elif GRANULE in ('03','04','05'):
        projection_flag = 'N'
    return projection_flag

# PURPOSE: find if segment crosses previously-known grounding line position
def read_grounded_ice(base_dir, HEM, VARIABLES=[0]):
    # reading grounded ice shapefile
    shape = fiona.open(os.path.join(base_dir,grounded_shapefile[HEM]))
    epsg = shape.crs['init']
    # reduce to variables of interest if specified
    shape_entities = [f for f in shape.values() if int(f['id']) in VARIABLES]
    # create list of polygons
    lines = []
    # extract the entities and assign by tile name
    for i,ent in enumerate(shape_entities):
        # extract coordinates for entity
        line_obj = shapely.geometry.LineString(ent['geometry']['coordinates'])
        lines.append(line_obj)
    # create shapely multilinestring object
    mline_obj = shapely.geometry.MultiLineString(lines)
    # close the shapefile
    shape.close()
    # return the line string object for the ice sheet
    return (mline_obj,epsg)

# PURPOSE: Find indices of common reference points between two lists
# Determines which along-track points correspond with the across-track
def common_reference_points(XT, AT):
    ind2 = np.squeeze([np.flatnonzero(AT == p) for p in XT])
    return ind2

# PURPOSE: compress complete list of valid indices into a set of ranges
def compress_list(i,n):
    for a,b in itertools.groupby(enumerate(i), lambda v: ((v[1]-v[0])//n)*n):
        group = list(map(operator.itemgetter(1),b))
        yield (group[0], group[-1])

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
    # CI1 = conf_interval(XI, PDF1/np.sum(PDF1), CONF)
    CI1 = 5e3
    CMN1,CMX1 = (XI[n]-CI1,XI[nn]+CI1)
    # CI2 = conf_interval(XI, PDF2/np.sum(PDF2), CONF)
    CI2 = 5e3
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
    # return the cutoff parameters, their confidence interval and the model
    return ([XI[n],CMN1,CMX1], [XI[nn],CMN2,CMX2], MODEL)

# PURPOSE: run a physical elastic bending model with Levenberg-Marquardt
# D. G. Vaughan, Journal of Geophysical Research Solid Earth, 1995
# A. M. Smith, Journal of Glaciology, 1991
def physical_elastic_model(XI,YI,METHOD='trf',GRZ=[0,0,0],TIDE=[0,0,0],
    ORIENTATION=False,THICKNESS=None,CONF=0.95,XOUT=None):
    # reorient input parameters to go from land ice to floating
    if XOUT is None:
        XOUT = np.copy(XI)
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
    popt,pcov = scipy.optimize.curve_fit(elasticmodel, XI, YI,
        p0=p0, bounds=bounds, method=METHOD)
    MODEL = elasticmodel(XOUT, *popt)
    # elasticmodel function outputs and 1 standard devation uncertainties
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
    return (dH + eta)

# PURPOSE: calculate the confidence interval in the retrieval
def conf_interval(x,f,p):
    # sorting probability distribution from smallest probability to largest
    ii = np.argsort(f)
    # compute the sorted cumulative probability distribution
    cdf = np.cumsum(f[ii])
    # linearly interpolate to confidence interval
    J = np.interp(p, cdf, x[ii])
    # position with maximum probability
    K = x[ii[-1]]
    return np.abs(K-J)

# PURPOSE: read ICESat-2 annual land ice height data (ATL11) from NSIDC
# calculate mean elevation between all dates in file
# calculate inflexion point using elevation surface slopes
# use mean elevation to calculate elevation anomalies
# use anomalies to calculate inward and seaward limits of tidal flexure
def calculate_GZ_ICESat2(base_dir, FILE, CROSSOVERS=False, TIDE_MODEL=None,
    REANALYSIS=None, PLOT=False, VERBOSE=False, MODE=0o775):
    # print file information
    print(os.path.basename(FILE)) if VERBOSE else None
    # read data from FILE
    mds1,attr1,pairs1 = read_HDF5_ATL11(FILE, REFERENCE=True,
        CROSSOVERS=CROSSOVERS, ATTRIBUTES=True, VERBOSE=VERBOSE)
    DIRECTORY = os.path.dirname(FILE)
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(FILE).pop()
    # file format for associated auxiliary files
    file_format = '{0}_{1}_{2}_{3}{4}_{5}{6}_{7}_{8}{9}.h5'
    # set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRAN)
    # grounded ice line string to determine if segment crosses coastline
    mline_obj,epsg = read_grounded_ice(base_dir, HEM)

    # height threshold (filter points below 0m elevation)
    THRESHOLD = 0.0
    # densities of seawater and ice
    rho_w = 1030.0
    rho_ice = 917.0

    # projections for converting lat/lon to polar stereographic
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    crs2 = pyproj.CRS.from_string(epsg)
    # transformer object for converting projections
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # copy variables for outputting to HDF5 file
    IS2_atl11_gz = {}
    IS2_atl11_fill = {}
    IS2_atl11_dims = {}
    IS2_atl11_gz_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl11_gz['ancillary_data'] = {}
    IS2_atl11_gz_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl11_gz['ancillary_data'][key] = mds1['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl11_gz_attrs['ancillary_data'][key] = {}
        for att_name,att_val in attr1['ancillary_data'][key].items():
            IS2_atl11_gz_attrs['ancillary_data'][key][att_name] = att_val
    # HDF5 group name for across-track data
    XT = 'crossing_track_data'
    # HDF5 group name for grounding zone data
    GZD = 'grounding_zone_data'
    GROUNDING_ZONE = True

    # for each input beam within the file
    for ptx in sorted(pairs1):
        # output data dictionaries for beam pair
        IS2_atl11_gz[ptx] = dict(cycle_stats=collections.OrderedDict(),
            crossing_track_data=collections.OrderedDict(),
            grounding_zone_data=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})
        IS2_atl11_dims[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})
        IS2_atl11_gz_attrs[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})

        # extract along-track and across-track variables
        ref_pt = {}
        latitude = {}
        longitude = {}
        delta_time = {}
        h_corr = {}
        tide_ocean = {}
        IB = {}
        groups = ['AT']
        # number of average segments and number of included cycles
        # fill_value for invalid heights and corrections
        fv = attr1[ptx]['h_corr']['_FillValue']
        # shape of along-track data
        n_points,n_cycles = mds1[ptx]['delta_time'].shape
        # along-track (AT) reference point, latitude, longitude and time
        ref_pt['AT'] = mds1[ptx]['ref_pt'].copy()
        latitude['AT'] = np.ma.array(mds1[ptx]['latitude'],
            fill_value=attr1[ptx]['latitude']['_FillValue'])
        latitude['AT'].mask = (latitude['AT'] == latitude['AT'].fill_value)
        longitude['AT'] = np.ma.array(mds1[ptx]['longitude'],
            fill_value=attr1[ptx]['longitude']['_FillValue'])
        longitude['AT'].mask = (longitude['AT'] == longitude['AT'].fill_value)
        delta_time['AT'] = np.ma.array(mds1[ptx]['delta_time'],
            fill_value=attr1[ptx]['delta_time']['_FillValue'])
        delta_time['AT'].mask = (delta_time['AT'] == delta_time['AT'].fill_value)
        # corrected height
        h_corr['AT'] = np.ma.array(mds1[ptx]['h_corr'],
            fill_value=attr1[ptx]['h_corr']['_FillValue'])
        h_corr['AT'].mask = (h_corr['AT'].data == h_corr['AT'].fill_value)
        # quality summary
        quality_summary = (mds1[ptx]['quality_summary'] == 0)
        # ocean corrections
        tide_ocean['AT'] = np.ma.array(mds1[ptx]['cycle_stats']['tide_ocean'],
            fill_value=attr1[ptx]['cycle_stats']['tide_ocean']['_FillValue'])
        tide_ocean['AT'].mask = (tide_ocean['AT'] == tide_ocean['AT'].fill_value)
        IB['AT'] = np.ma.array(mds1[ptx]['cycle_stats']['dac'],
            fill_value=attr1[ptx]['cycle_stats']['dac']['_FillValue'])
        IB['AT'].mask = (IB['AT'] == IB['AT'].fill_value)
        # ATL11 reference surface elevations (derived from ATL06)
        dem_h = mds1[ptx]['ref_surf']['dem_h']
        # geoid_h = mds1[ptx]['ref_surf']['geoid_h']
        # if running ATL11 crossovers
        if CROSSOVERS:
            # add to group
            groups.append('XT')
            # shape of across-track data
            n_cross, = mds1[ptx][XT]['delta_time'].shape
            # across-track (XT) reference point, latitude, longitude and time
            ref_pt['XT'] = mds1[ptx][XT]['ref_pt'].copy()
            latitude['XT'] = np.ma.array(mds1[ptx][XT]['latitude'],
                fill_value=attr1[ptx][XT]['latitude']['_FillValue'])
            latitude['XT'].mask = (latitude['XT'] == latitude['XT'].fill_value)
            longitude['XT'] = np.ma.array(mds1[ptx][XT]['longitude'],
                fill_value=attr1[ptx][XT]['longitude']['_FillValue'])
            latitude['XT'].mask = (latitude['XT'] == longitude['XT'].fill_value)
            delta_time['XT'] = np.ma.array(mds1[ptx][XT]['delta_time'],
                fill_value=attr1[ptx][XT]['delta_time']['_FillValue'])
            delta_time['XT'].mask = (delta_time['XT'] == delta_time['XT'].fill_value)
            # corrected height at crossovers
            h_corr['XT'] = np.ma.array(mds1[ptx][XT]['h_corr'],
                fill_value=attr1[ptx][XT]['h_corr']['_FillValue'])
            h_corr['XT'].mask = (h_corr['XT'].data == h_corr['XT'].fill_value)
            # across-track (XT) ocean corrections
            tide_ocean['XT'] = np.ma.array(mds1[ptx][XT]['tide_ocean'],
                fill_value=attr1[ptx][XT]['tide_ocean']['_FillValue'])
            tide_ocean['XT'].mask = (tide_ocean['XT'] == tide_ocean['XT'].fill_value)
            IB['XT'] = np.ma.array(mds1[ptx][XT]['dac'],
                fill_value=attr1[ptx][XT]['dac']['_FillValue'])
            IB['XT'].mask = (IB['XT'] == IB['XT'].fill_value)

        # read buffered grounding zone mask
        a2 = (PRD,'GROUNDING_ZONE','MASK',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        f3 = os.path.join(DIRECTORY,file_format.format(*a2))
        # create data mask for grounding zone
        mds1[ptx]['subsetting'] = {}
        mds1[ptx]['subsetting'].setdefault('ice_gz',
            np.zeros((n_points),dtype=bool))
        # check that mask file exists
        try:
            mds2,attr2 = read_HDF5_ATL11_pair(f3,ptx,
                ATTRIBUTES=True,VERBOSE=False,SUBSETTING=True)
        except:
            pass
        else:
            mds1[ptx]['subsetting']['ice_gz'] = \
                mds2[ptx]['subsetting']['ice_gz']
            B = attr2[ptx]['subsetting']['ice_gz']['source']

        # read tide model
        if TIDE_MODEL:
            # read tide model HDF5 file
            a3 = (PRD,TIDE_MODEL,'TIDES',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            f3 = os.path.join(DIRECTORY,file_format.format(*a3))
            # check that tide model file exists
            try:
                mds3,attr3 = read_HDF5_ATL11_pair(f3,ptx,
                    VERBOSE=False,CROSSOVERS=CROSSOVERS)
            except:
                # mask all values
                for group in groups:
                    tide_ocean[group].mask[:] = True
                pass
            else:
                tide_ocean['AT'].data[:] = mds3[ptx]['cycle_stats']['tide_ocean']
                if CROSSOVERS:
                    tide_ocean['XT'].data[:] = mds3[ptx][XT]['tide_ocean']
            # source of tide model
            tide_source = TIDE_MODEL
        else:
            tide_source = 'ATL06'
        # set masks and fill values
        for group,val in tide_ocean.items():
            val.mask[:] = (val.data == val.fill_value)
            val.mask[:] |= (h_corr[group].data == h_corr[group].fill_value)
            val.data[val.mask] = val.fill_value

        # read inverse barometer correction
        if REANALYSIS:
            # read inverse barometer HDF5 file
            a4 = (PRD,REANALYSIS,'IB',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            f4 = os.path.join(DIRECTORY,file_format.format(*a4))
            # check that inverse barometer file exists
            try:
                mds4,attr4 = read_HDF5_ATL11_pair(f4,ptx,
                    VERBOSE=False,CROSSOVERS=CROSSOVERS)
            except:
                # mask all values
                for group in groups:
                    IB[group].mask[:] = True
                pass
            else:
                IB['AT'].data[:] = mds4[ptx]['cycle_stats']['ib']
                if CROSSOVERS:
                    IB['XT'].data[:] = mds4[ptx][XT]['ib']
        # set masks and fill values
        for group,val in IB.items():
            val.mask[:] = (val.data == val.fill_value)
            val.mask[:] |= (h_corr[group].data == h_corr[group].fill_value)
            val.data[val.mask] = val.fill_value

        # mean dynamic topography
        a5 = (PRD,'AVISO','SEA_LEVEL',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        f5 = os.path.join(DIRECTORY,file_format.format(*a5))
        # check that mean dynamic topography file exists
        try:
            mds5,attr5 = read_HDF5_ATL11_pair(f5,ptx,VERBOSE=False)
        except:
            mdt = np.zeros((n_points))
            pass
        else:
            mdt = mds5[ptx]['cycle_stats']['mdt']

        # extract lat/lon and convert to polar stereographic
        X,Y = transformer.transform(longitude['AT'],latitude['AT'])
        # along-track (AT) flexure corrections
        fv = attr1[ptx]['cycle_stats']['tide_ocean']['_FillValue']
        flexure = np.ma.zeros((n_points,n_cycles),fill_value=fv)
        # initally copy the ocean tide estimate
        flexure.data[:] = np.copy(tide_ocean['AT'].data)
        flexure.mask = np.copy(tide_ocean['AT'].mask)
        # scaling factor for segment tides
        scaling = np.ma.ones((n_points,n_cycles),fill_value=0.0)
        scaling.mask = np.copy(tide_ocean['AT'].mask)
        scaling.data[scaling.mask]

        # outputs of grounding zone fit
        grounding_zone_data = {}
        grounding_zone_data['ref_pt'] = []
        grounding_zone_data['latitude'] = []
        grounding_zone_data['longitude'] = []
        grounding_zone_data['delta_time'] = []
        grounding_zone_data['cycle_number'] = []
        # grounding_zone_data['tide_ocean'] = []
        grounding_zone_data['gz_sigma'] = []
        grounding_zone_data['e_mod'] = []
        grounding_zone_data['e_mod_sigma'] = []
        # grounding_zone_data['H_ice'] = []
        # grounding_zone_data['delta_h'] = []

        # if creating a test plot
        valid_plot = False
        if PLOT:
            f1,ax1 = plt.subplots(num=1,figsize=(13,7))

        # for each cycle of ATL11 data
        for c,CYCLE in enumerate(mds1[ptx]['cycle_number']):
            # find valid points with GZ for any ATL11 cycle
            segment_mask = np.logical_not(h_corr['AT'].mask[:,c])
            segment_mask = np.logical_not(tide_ocean['AT'].mask[:,c])
            segment_mask &= (h_corr['AT'].data[:,c] > THRESHOLD)
            segment_mask &= mds1[ptx]['subsetting']['ice_gz']
            segment_mask &= quality_summary[:,c]
            ifit, = np.nonzero(segment_mask)
            # segment of points within grounding zone
            igz, = np.nonzero(mds1[ptx]['subsetting']['ice_gz'])

            # compress list (separate geosegs into sets of ranges)
            ice_gz_indices = compress_list(ifit,1000)
            for imin,imax in ice_gz_indices:
                # find valid indices within range
                i = sorted(set(np.arange(imin,imax+1)) & set(ifit))
                iout = sorted(set(np.arange(imin,imax+1)) & set(igz))
                coords = np.sqrt((X-X[i[0]])**2 + (Y-Y[i[0]])**2)
                # shapely LineString object for altimetry segment
                try:
                    segment_line = shapely.geometry.LineString(np.c_[X[i],Y[i]])
                except:
                    continue
                # determine if line segment intersects previously known GZ
                if segment_line.intersects(mline_obj):
                    # extract intersected point (find minimum distance)
                    try:
                        xi,yi = mline_obj.intersection(segment_line).xy
                    except:
                        continue
                    else:
                        iint = np.argmin((Y[i]-yi)**2 + (X[i]-xi)**2)
                    # horizontal eulerian distance from start of segment
                    dist = coords[i]
                    output = coords[iout]
                    # land ice height for grounding zone
                    h_gz = np.copy(h_corr['AT'].data[i,c])
                    # mean land ice height from digital elevation model
                    h_mean = np.mean(h_corr['AT'][i,:],axis=1)
                    # h_mean = h_corr['AT'].data[i,0]
                    # ocean tide height for scaling model
                    tide_mean =  np.mean(tide_ocean['AT'][i,:],axis=1)
                    # tide_mean = tide_ocean['AT'].data[i,0]
                    h_tide = np.ma.array(tide_ocean['AT'].data[i,c] - tide_mean,
                        fill_value=tide_ocean['AT'].fill_value)
                    h_tide.mask = tide_ocean['AT'].mask[i,c] | tide_mean.mask
                    # inverse-barometer response
                    ib_mean =  np.mean(IB['AT'][i,:],axis=1)
                    # ib_mean = IB['AT'].data[i,0]
                    h_ib = np.ma.array(IB['AT'].data[i,c] - ib_mean,
                        fill_value=IB['AT'].fill_value)
                    h_ib.mask = IB['AT'].mask[i,c] | ib_mean.mask
                    # deflection from mean land ice height in grounding zone
                    dh_gz = h_gz - h_mean
                    # quasi-freeboard: WGS84 elevation - geoid height
                    QFB = h_gz #- geoid_h
                    # ice thickness from quasi-freeboard and densities
                    w_thick = QFB*rho_w/(rho_w-rho_ice)
                    # fit with a hard piecewise model to get rough estimate of GZ
                    try:
                        C1,C2,PWMODEL = piecewise_fit(dist, dh_gz, STEP=5, CONF=0.95)
                    except:
                        continue
                    # distance from estimated grounding line (0 = grounding line)
                    d = (dist - C1[0]).astype(int)
                    # determine if spacecraft is approaching coastline
                    sco = True if np.mean(h_gz[d<0]) < np.mean(h_gz[d>0]) else False
                    # set initial fit outputs to infinite
                    PGZ = np.array([np.inf,np.inf])
                    # set grounding zone estimates for testing
                    GRZ = []
                    # 1,2: use GZ location values from piecewise fit
                    # 3,4: use GZ location values from known grounding line
                    GRZ.append(C1)
                    GRZ.append(C1)
                    GRZ.append([dist[iint],dist[iint]-2e3,dist[iint]+2e3])
                    GRZ.append([dist[iint],dist[iint]-2e3,dist[iint]+2e3])
                    # set tide values for testing
                    TIDE = []
                    i0 = 0 if sco else -1
                    tplus = h_tide[i0] + h_ib[i0]
                    # 1,3: use tide range values from Padman (2002)
                    # 2,4: use tide range values from model+ib
                    TIDE.append([1.2,-3.0,3.0])
                    TIDE.append([tplus,tplus-0.3,tplus+0.3])
                    TIDE.append([1.2,-3.0,3.0])
                    TIDE.append([tplus,tplus-0.3,tplus+0.3])
                    # iterate through tests
                    for grz,tide in zip(GRZ,TIDE):
                        # fit physical elastic model
                        try:
                            GZ,PA,PE,PT,PdH,MODEL = physical_elastic_model(dist,
                                dh_gz, GRZ=grz, TIDE=tide, ORIENTATION=sco,
                                THICKNESS=w_thick, CONF=0.95, XOUT=output)
                        except:
                            pass
                        # copy grounding zone parameters to get best fit
                        if (GZ[1] < PGZ[1]):
                            PGZ = np.copy(GZ)
                            model_scale = np.copy(PA[0])
                            PEMODEL = np.copy(MODEL)
                        # use parameters if fit significance is within tolerance
                        if (GZ[1] < 400.0):
                            break

                    # linearly interpolate distance to grounding line
                    GZrpt = np.interp(PGZ[0],output,ref_pt['AT'][iout])
                    GZlat = np.interp(PGZ[0],output,latitude['AT'][iout])
                    GZlon = np.interp(PGZ[0],output,longitude['AT'][iout])
                    GZtime = np.interp(PGZ[0],dist,delta_time['AT'][i,c])
                    # append outputs of grounding zone fit
                    # save all outputs (not just within tolerance)
                    grounding_zone_data['ref_pt'].append(GZrpt)
                    grounding_zone_data['latitude'].append(GZlat)
                    grounding_zone_data['longitude'].append(GZlon)
                    grounding_zone_data['delta_time'].append(GZtime)
                    grounding_zone_data['cycle_number'].append(CYCLE)
                    # grounding_zone_data['tide_ocean'].append(PA)
                    grounding_zone_data['gz_sigma'].append(PGZ[1])
                    grounding_zone_data['e_mod'].append(PE[0]/1e9)
                    grounding_zone_data['e_mod_sigma'].append(PE[1]/1e9)
                    # grounding_zone_data['H_ice'].append(PT)
                    # grounding_zone_data['delta_h'].append(PdH)

                    # reorient input parameters to go from land ice to floating
                    flexure_mask = np.ones_like(iout,dtype=bool)
                    if sco:
                        # start of segment in orientation
                        i0 = iout[0]
                        # mean tide for scaling and plots
                        # mean_tide = tide_ocean['AT'].data[i0,0]
                        mean_tide = np.mean(tide_ocean['AT'][i0,:])
                        mean_ib = np.mean(IB['AT'][i0,:])
                        tide_scale = tide_ocean['AT'].data[i0,c] - mean_tide
                        # replace mask values for points beyond the grounding line
                        ii, = np.nonzero(ref_pt['AT'][iout] <= GZrpt)
                        flexure_mask[ii] = False
                    else:
                        # start of segment in orientation
                        i0 = iout[-1]
                        # mean tide for scaling and plots
                        # mean_tide = tide_ocean['AT'].data[i0,0]
                        mean_tide = np.mean(tide_ocean['AT'][i0,:])
                        mean_ib = np.mean(IB['AT'][i0,:])
                        tide_scale = tide_ocean['AT'].data[i0,c] - mean_tide
                        # replace mask values for points beyond the grounding line
                        ii, = np.nonzero(ref_pt['AT'][iout] >= GZrpt)
                        flexure_mask[ii] = False
                    # add to test plot
                    if PLOT:
                        # plot height differences
                        l, = ax1.plot(ref_pt['AT'][i],dh_gz-PdH[0],'.-',ms=1.5,lw=0,
                            label='Cycle {0}'.format(CYCLE))
                        # plot downstream tide and IB
                        hocean = tide_ocean['AT'].data[i0,c] - mean_tide
                        # hocean += IB['AT'].data[i0,c] - mean_ib
                        ax1.axhline(hocean,color=l.get_color(),lw=3.0,ls='--')
                        # set valid plot flag
                        valid_plot = True

                    # if the grounding zone errors are not within tolerance
                    if (PGZ[1] >= 800.0):
                        # leave iteration and keep original tide model
                        # for segment
                        continue

                    # calculate scaling factor
                    scale_factor = tide_scale/model_scale
                    # scale flexure and restore mean ocean tide
                    flexure[iout,c] = scale_factor*(PEMODEL-PdH[0]) + mean_tide
                    flexure.mask[iout,c] = flexure_mask
                    # scaling factor between current tide and flexure
                    scaling[iout,c] = flexure[iout,c]/tide_ocean['AT'].data[iout,c]
                    scaling.mask[iout,c] = flexure_mask
                    # add to test plot
                    if PLOT:
                        # plot elastic deformation model
                        ax1.plot(ref_pt['AT'][iout],PEMODEL-PdH[0],
                            color='0.3',lw=2,zorder=9)
                        # plot scaled elastic deformation model
                        ax1.plot(ref_pt['AT'][iout],flexure[iout,c]-mean_tide,
                            color='0.8',lw=2,zorder=10)
                        # plot grounding line location
                        ax1.axvline(GZrpt,color=l.get_color(),
                            ls='--',dashes=(8,4))

        # make final plot adjustments and save to file
        if valid_plot:
            # add legend
            lgd = ax1.legend(loc=1,frameon=True)
            # set width, color and style of lines
            lgd.get_frame().set_boxstyle('square,pad=0.1')
            lgd.get_frame().set_edgecolor('white')
            lgd.get_frame().set_alpha(1.0)
            for line,text in zip(lgd.get_lines(),lgd.get_texts()):
                line.set_linewidth(6)
                text.set_weight('bold')
                text.set_color(line.get_color())
            # adjust figure
            f1.subplots_adjust(left=0.05,right=0.97,bottom=0.04,top=0.96,hspace=0.15)
            # create plot file of flexural zone
            args = (PRD,ptx,TIDE_MODEL,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            plot_format = '{0}_{1}_{2}_GZ_TIDES_{3}{4}_{5}{6}_{7}_{8}{9}.png'
            f1.savefig(os.path.join(DIRECTORY,plot_format.format(*args)), dpi=240,
                metadata={'Title':os.path.basename(sys.argv[0])}, format='png')
            # clear all figure axes
            plt.cla()
            plt.clf()

        # group attributes for beam
        IS2_atl11_gz_attrs[ptx]['description'] = ('Contains the primary science parameters '
            'for this data set')
        IS2_atl11_gz_attrs[ptx]['beam_pair'] = attr1[ptx]['beam_pair']
        IS2_atl11_gz_attrs[ptx]['ReferenceGroundTrack'] = attr1[ptx]['ReferenceGroundTrack']
        IS2_atl11_gz_attrs[ptx]['first_cycle'] = attr1[ptx]['first_cycle']
        IS2_atl11_gz_attrs[ptx]['last_cycle'] = attr1[ptx]['last_cycle']
        IS2_atl11_gz_attrs[ptx]['equatorial_radius'] = attr1[ptx]['equatorial_radius']
        IS2_atl11_gz_attrs[ptx]['polar_radius'] = attr1[ptx]['polar_radius']

        # geolocation, time and reference point
        # reference point
        IS2_atl11_gz[ptx]['ref_pt'] = ref_pt['AT'].copy()
        IS2_atl11_fill[ptx]['ref_pt'] = None
        IS2_atl11_dims[ptx]['ref_pt'] = None
        IS2_atl11_gz_attrs[ptx]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['ref_pt']['units'] = "1"
        IS2_atl11_gz_attrs[ptx]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_gz_attrs[ptx]['ref_pt']['long_name'] = "Reference point number"
        IS2_atl11_gz_attrs[ptx]['ref_pt']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['ref_pt']['description'] = ("The reference point is the "
            "7 digit segment_id number corresponding to the center of the ATL06 data used "
            "for each ATL11 point.  These are sequential, starting with 1 for the first "
            "segment after an ascending equatorial crossing node.")
        IS2_atl11_gz_attrs[ptx]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number
        IS2_atl11_gz[ptx]['cycle_number'] = mds1[ptx]['cycle_number'].copy()
        IS2_atl11_fill[ptx]['cycle_number'] = None
        IS2_atl11_dims[ptx]['cycle_number'] = None
        IS2_atl11_gz_attrs[ptx]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['cycle_number']['units'] = "1"
        IS2_atl11_gz_attrs[ptx]['cycle_number']['long_name'] = "Orbital cycle number"
        IS2_atl11_gz_attrs[ptx]['cycle_number']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['cycle_number']['description'] = ("Number of 91-day periods "
            "that have elapsed since ICESat-2 entered the science orbit. Each of the 1,387 "
            "reference ground track (RGTs) is targeted in the polar regions once "
            "every 91 days.")
        # delta time
        IS2_atl11_gz[ptx]['delta_time'] = delta_time['AT'].copy()
        IS2_atl11_fill[ptx]['delta_time'] = delta_time['AT'].fill_value
        IS2_atl11_dims[ptx]['delta_time'] = ['ref_pt','cycle_number']
        IS2_atl11_gz_attrs[ptx]['delta_time'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_gz_attrs[ptx]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_gz_attrs[ptx]['delta_time']['standard_name'] = "time"
        IS2_atl11_gz_attrs[ptx]['delta_time']['calendar'] = "standard"
        IS2_atl11_gz_attrs[ptx]['delta_time']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_gz_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt cycle_number latitude longitude"
        # latitude
        IS2_atl11_gz[ptx]['latitude'] = latitude['AT'].copy()
        IS2_atl11_fill[ptx]['latitude'] = latitude['AT'].fill_value
        IS2_atl11_dims[ptx]['latitude'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx]['latitude'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['latitude']['units'] = "degrees_north"
        IS2_atl11_gz_attrs[ptx]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx]['latitude']['long_name'] = "Latitude"
        IS2_atl11_gz_attrs[ptx]['latitude']['standard_name'] = "latitude"
        IS2_atl11_gz_attrs[ptx]['latitude']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_gz_attrs[ptx]['latitude']['valid_min'] = -90.0
        IS2_atl11_gz_attrs[ptx]['latitude']['valid_max'] = 90.0
        IS2_atl11_gz_attrs[ptx]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude
        IS2_atl11_gz[ptx]['longitude'] = longitude['AT'].copy()
        IS2_atl11_fill[ptx]['longitude'] = longitude['AT'].fill_value
        IS2_atl11_dims[ptx]['longitude'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx]['longitude'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['longitude']['units'] = "degrees_east"
        IS2_atl11_gz_attrs[ptx]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx]['longitude']['long_name'] = "Longitude"
        IS2_atl11_gz_attrs[ptx]['longitude']['standard_name'] = "longitude"
        IS2_atl11_gz_attrs[ptx]['longitude']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_gz_attrs[ptx]['longitude']['valid_min'] = -180.0
        IS2_atl11_gz_attrs[ptx]['longitude']['valid_max'] = 180.0
        IS2_atl11_gz_attrs[ptx]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"

        # cycle statistics variables
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['Description'] = ("The cycle_stats subgroup "
            "contains summary information about segments for each reference point, including "
            "the uncorrected mean heights for reference surfaces, blowing snow and cloud "
            "indicators, and geolocation and height misfit statistics.")
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['data_rate'] = ("Data within this group "
            "are stored at the average segment rate.")
        # computed tide with flexure
        flexure.data[flexure.mask] = flexure.fill_value
        IS2_atl11_gz[ptx]['cycle_stats']['tide_ocean'] = flexure.copy()
        IS2_atl11_fill[ptx]['cycle_stats']['tide_ocean'] = flexure.fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['tide_ocean'] = ['ref_pt','cycle_number']
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['units'] = "meters"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['contentType'] = "referenceInformation"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['long_name'] = "Ocean Tide"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['description'] = ("Ocean Tides with "
            "Near-Grounding Zone Flexure that includes diurnal and semi-diurnal (harmonic analysis), "
            "and longer period tides (dynamic and self-consistent equilibrium).")
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['source'] = tide_source
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['reference'] = \
            "https://doi.org/10.3189/172756410791392790"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"
        # ratio of flexure with respect to downstream ocean tide
        scaling.data[scaling.mask] = scaling.fill_value
        IS2_atl11_gz[ptx]['cycle_stats']['flexure'] = scaling.copy()
        IS2_atl11_fill[ptx]['cycle_stats']['flexure'] = scaling.fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['flexure'] = ['ref_pt','cycle_number']
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['units'] = "1"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['contentType'] = "referenceInformation"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['long_name'] = "Flexure Ratio"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['description'] = ("Ratio of "
            "Near-Grounding Zone Flexure with respect to Downstream Ocean Tide Height.")
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['source'] = tide_source
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['reference'] = \
            "https://doi.org/10.3189/172756410791392790"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"

        # grounding zone variables
        IS2_atl11_gz_attrs[ptx][GZD]['Description'] = ("The grounding_zone_data "
            "subgroup contains statistic data at grounding zone locations.")
        IS2_atl11_gz_attrs[ptx][GZD]['data_rate'] = ("Data within this group are "
            "stored at the average segment rate.")

        # reference point of the grounding zone
        IS2_atl11_gz[ptx][GZD]['ref_pt'] = np.copy(grounding_zone_data['ref_pt'])
        IS2_atl11_fill[ptx][GZD]['ref_pt'] = None
        IS2_atl11_dims[ptx][GZD]['ref_pt'] = None
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['units'] = "1"
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['long_name'] = ("fit center reference point number, "
            "segment_id")
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['source'] = "derived, ATL11 algorithm"
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['description'] = ("The reference-point number of the "
            "fit center for the datum track. The reference point is the 7 digit segment_id number "
            "corresponding to the center of the ATL06 data used for each ATL11 point.  These are "
            "sequential, starting with 1 for the first segment after an ascending equatorial "
            "crossing node.")
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number of the grounding zone
        IS2_atl11_gz[ptx][GZD]['cycle_number'] = np.copy(grounding_zone_data['cycle_number'])
        IS2_atl11_fill[ptx][GZD]['cycle_number'] = None
        IS2_atl11_dims[ptx][GZD]['cycle_number'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number']['units'] = "1"
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number']['long_name'] = "cycle number"
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number']['description'] = ("Cycle number for the "
            "grounding zone data. Number of 91-day periods that have elapsed since ICESat-2 entered "
            "the science orbit. Each of the 1,387 reference ground track (RGTs) is targeted "
            "in the polar regions once every 91 days.")
        # delta time of the grounding zone
        IS2_atl11_gz[ptx][GZD]['delta_time'] = np.copy(grounding_zone_data['delta_time'])
        IS2_atl11_fill[ptx][GZD]['delta_time'] = delta_time['AT'].fill_value
        IS2_atl11_dims[ptx][GZD]['delta_time'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time'] = {}
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['standard_name'] = "time"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['calendar'] = "standard"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['coordinates'] = \
            "ref_pt latitude longitude"
        # latitude of the grounding zone
        IS2_atl11_gz[ptx][GZD]['latitude'] = np.copy(grounding_zone_data['latitude'])
        IS2_atl11_fill[ptx][GZD]['latitude'] = latitude['AT'].fill_value
        IS2_atl11_dims[ptx][GZD]['latitude'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['latitude'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['units'] = "degrees_north"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['long_name'] = "grounding zone latitude"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['standard_name'] = "latitude"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['description'] = ("Center latitude of "
            "the grounding zone")
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['valid_min'] = -90.0
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['valid_max'] = 90.0
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude of the grounding zone
        IS2_atl11_gz[ptx][GZD]['longitude'] = np.copy(grounding_zone_data['longitude'])
        IS2_atl11_fill[ptx][GZD]['longitude'] = longitude['AT'].fill_value
        IS2_atl11_dims[ptx][GZD]['longitude'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['longitude'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['units'] = "degrees_east"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['long_name'] = "grounding zone longitude"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['standard_name'] = "longitude"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['description'] = ("Center longitude of "
            "the grounding zone")
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['valid_min'] = -180.0
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['valid_max'] = 180.0
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"
        # uncertainty of the grounding zone
        IS2_atl11_gz[ptx][GZD]['gz_sigma'] = np.copy(grounding_zone_data['gz_sigma'])
        IS2_atl11_fill[ptx][GZD]['gz_sigma'] = 0.0
        IS2_atl11_dims[ptx][GZD]['gz_sigma'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['units'] = "meters"
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['long_name'] = "grounding zone uncertainty"
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['source'] = "ATL11"
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['description'] = ("Uncertainty in grounding"
            "zone location derived by the physical elastic bending model")
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # effective elastic modulus
        IS2_atl11_gz[ptx][GZD]['e_mod'] = np.copy(grounding_zone_data['e_mod'])
        IS2_atl11_fill[ptx][GZD]['e_mod'] = 0.0
        IS2_atl11_dims[ptx][GZD]['e_mod'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['units'] = "GPa"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['long_name'] = "Elastic modulus"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['source'] = "ATL11"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['description'] = ("Effective Elastic modulus "
            "of ice estimating using an elastic beam model")
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # uncertainty of the elastic modulus
        IS2_atl11_gz[ptx][GZD]['e_mod_sigma'] = np.copy(grounding_zone_data['e_mod_sigma'])
        IS2_atl11_fill[ptx][GZD]['e_mod_sigma'] = 0.0
        IS2_atl11_dims[ptx][GZD]['e_mod_sigma'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['units'] = "GPa"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['long_name'] = "Elastic modulus uncertainty"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['source'] = "ATL11"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['description'] = ("Uncertainty in the "
            "effective Elastic modulus of ice")
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['coordinates'] = \
            "ref_pt delta_time latitude longitude"

        # if estimating flexure for crossover measurements
        if CROSSOVERS:
            # calculate mean scaling for crossovers
            mean_scale = np.ma.zeros((n_points),fill_value=scaling.fill_value)
            mean_scale.data[:] = scaling.mean(axis=1)
            mean_scale.mask = np.all(scaling.mask,axis=1)
            # find mapping between crossover and along-track reference points
            ref_indices = common_reference_points(ref_pt['XT'], ref_pt['AT'])
            # scale input tide model for estimated flexure in region
            scaled_tide = np.ma.zeros((n_cross),fill_value=tide_ocean['XT'].fill_value)
            scaled_tide.data[:] = tide_ocean['XT']*mean_scale[ref_indices]
            scaled_tide.mask = mean_scale.mask[ref_indices]

            # crossing track variables
            IS2_atl11_gz_attrs[ptx][XT]['Description'] = ("The crossing_track_data "
                "subgroup contains elevation data at crossover locations. These are "
                "locations where two ICESat-2 pair tracks cross, so data are available "
                "from both the datum track, for which the granule was generated, and "
                "from the crossing track.")
            IS2_atl11_gz_attrs[ptx][XT]['data_rate'] = ("Data within this group are "
                "stored at the average segment rate.")

            # reference point
            IS2_atl11_gz[ptx][XT]['ref_pt'] = mds1[ptx][XT]['ref_pt'].copy()
            IS2_atl11_fill[ptx][XT]['ref_pt'] = None
            IS2_atl11_dims[ptx][XT]['ref_pt'] = None
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['units'] = "1"
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['contentType'] = "referenceInformation"
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['long_name'] = ("fit center reference point number, "
                "segment_id")
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['source'] = "derived, ATL11 algorithm"
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['description'] = ("The reference-point number of the "
                "fit center for the datum track. The reference point is the 7 digit segment_id number "
                "corresponding to the center of the ATL06 data used for each ATL11 point.  These are "
                "sequential, starting with 1 for the first segment after an ascending equatorial "
                "crossing node.")
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['coordinates'] = \
                "delta_time latitude longitude"
            # reference ground track of the crossing track
            IS2_atl11_gz[ptx][XT]['rgt'] = mds1[ptx][XT]['rgt'].copy()
            IS2_atl11_fill[ptx][XT]['rgt'] = attr1[ptx][XT]['rgt']['_FillValue']
            IS2_atl11_dims[ptx][XT]['rgt'] = None
            IS2_atl11_gz_attrs[ptx][XT]['rgt'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['units'] = "1"
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['contentType'] = "referenceInformation"
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['long_name'] = "crossover reference ground track"
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['description'] = "The RGT number for the crossing data."
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['coordinates'] = \
                "ref_pt delta_time latitude longitude"
            # cycle_number of the crossing track
            IS2_atl11_gz[ptx][XT]['cycle_number'] = mds1[ptx][XT]['cycle_number'].copy()
            IS2_atl11_fill[ptx][XT]['cycle_number'] = attr1[ptx][XT]['cycle_number']['_FillValue']
            IS2_atl11_dims[ptx][XT]['cycle_number'] = None
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number']['units'] = "1"
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number']['long_name'] = "crossover cycle number"
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number']['description'] = ("Cycle number for the "
                "crossing data. Number of 91-day periods that have elapsed since ICESat-2 entered "
                "the science orbit. Each of the 1,387 reference ground track (RGTs) is targeted "
                "in the polar regions once every 91 days.")
            # delta time of the crossing track
            IS2_atl11_gz[ptx][XT]['delta_time'] = delta_time['XT'].copy()
            IS2_atl11_fill[ptx][XT]['delta_time'] = delta_time['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['delta_time'] = ['ref_pt']
            IS2_atl11_gz_attrs[ptx][XT]['delta_time'] = {}
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['units'] = "seconds since 2018-01-01"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['long_name'] = "Elapsed GPS seconds"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['standard_name'] = "time"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['calendar'] = "standard"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['description'] = ("Number of GPS "
                "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
                "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
                "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
                "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
                "time in gps_seconds relative to the GPS epoch can be computed.")
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['coordinates'] = \
                "ref_pt latitude longitude"
            # latitude of the crossover measurement
            IS2_atl11_gz[ptx][XT]['latitude'] = latitude['XT'].copy()
            IS2_atl11_fill[ptx][XT]['latitude'] = latitude['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['latitude'] = ['ref_pt']
            IS2_atl11_gz_attrs[ptx][XT]['latitude'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['units'] = "degrees_north"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['contentType'] = "physicalMeasurement"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['long_name'] = "crossover latitude"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['standard_name'] = "latitude"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['description'] = ("Center latitude of "
                "selected segments")
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['valid_min'] = -90.0
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['valid_max'] = 90.0
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['coordinates'] = \
                "ref_pt delta_time longitude"
            # longitude of the crossover measurement
            IS2_atl11_gz[ptx][XT]['longitude'] = longitude['XT'].copy()
            IS2_atl11_fill[ptx][XT]['longitude'] = longitude['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['longitude'] = ['ref_pt']
            IS2_atl11_gz_attrs[ptx][XT]['longitude'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['units'] = "degrees_east"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['contentType'] = "physicalMeasurement"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['long_name'] = "crossover longitude"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['standard_name'] = "longitude"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['description'] = ("Center longitude of "
                "selected segments")
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['valid_min'] = -180.0
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['valid_max'] = 180.0
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['coordinates'] = \
                "ref_pt delta_time latitude"
            # computed tide with flexure for the crossover measurement
            IS2_atl11_gz[ptx][XT]['tide_ocean'] = scaled_tide.copy()
            IS2_atl11_fill[ptx][XT]['tide_ocean'] = scaled_tide.fill_value
            IS2_atl11_dims[ptx][XT]['tide_ocean'] = ['ref_pt']
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['units'] = "meters"
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['contentType'] = "referenceInformation"
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['long_name'] = "Ocean Tide"
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['description'] = ("Ocean Tides with "
                "Near-Grounding Zone Flexure that includes diurnal and semi-diurnal (harmonic analysis), "
                "and longer period tides (dynamic and self-consistent equilibrium).")
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['source'] = tide_source
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['reference'] = \
                "https://doi.org/10.3189/172756410791392790"
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['coordinates'] = \
                "ref_pt delta_time latitude longitude"

    # output flexure correction HDF5 file
    args = (PRD,TIDE_MODEL,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
    file_format = '{0}_{1}_GZ_TIDES_{2}{3}_{4}{5}_{6}_{7}{8}.h5'
    # print file information
    print('\t{0}'.format(file_format.format(*args))) if VERBOSE else None
    HDF5_ATL11_corr_write(IS2_atl11_gz, IS2_atl11_gz_attrs,
        CLOBBER=True, INPUT=os.path.basename(FILE),
        GROUNDING_ZONE=GROUNDING_ZONE, CROSSOVERS=CROSSOVERS,
        FILL_VALUE=IS2_atl11_fill, DIMENSIONS=IS2_atl11_dims,
        FILENAME=os.path.join(DIRECTORY,file_format.format(*args)))
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,file_format.format(*args)), MODE)

# PURPOSE: outputting the correction values for ICESat-2 data to HDF5
def HDF5_ATL11_corr_write(IS2_atl11_corr, IS2_atl11_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, GROUNDING_ZONE=False,
    CROSSOVERS=False, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    # create HDF5 records
    h5 = {}

    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl11_corr['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam pair
    pairs = [k for k in IS2_atl11_corr.keys() if bool(re.match(r'pt\d',k))]
    for ptx in pairs:
        fileID.create_group(ptx)
        h5[ptx] = {}
        # add HDF5 group attributes for beam
        for att_name in ['description','beam_pair','ReferenceGroundTrack',
            'first_cycle','last_cycle','equatorial_radius','polar_radius']:
            fileID[ptx].attrs[att_name] = IS2_atl11_attrs[ptx][att_name]

        # ref_pt, cycle number, geolocation and delta_time variables
        for k in ['ref_pt','cycle_number','delta_time','latitude','longitude']:
            # values and attributes
            v = IS2_atl11_corr[ptx][k]
            attrs = IS2_atl11_attrs[ptx][k]
            fillvalue = FILL_VALUE[ptx][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}'.format(ptx,k)
            if fillvalue:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[ptx][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[ptx][k]):
                    h5[ptx][k].dims[i].attach_scale(h5[ptx][dim])
            else:
                # make dimension
                h5[ptx][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[ptx][k].attrs[att_name] = att_val

        # add to cycle_stats variables
        groups = ['cycle_stats']
        # if there were valid fits: add to grounding_zone_data variables
        if GROUNDING_ZONE:
            groups.append('grounding_zone_data')
        # if running crossovers: add to crossing_track_data variables
        if CROSSOVERS:
            groups.append('crossing_track_data')
        for key in groups:
            fileID[ptx].create_group(key)
            h5[ptx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl11_attrs[ptx][key][att_name]
                fileID[ptx][key].attrs[att_name] = att_val
            for k,v in IS2_atl11_corr[ptx][key].items():
                # attributes
                attrs = IS2_atl11_attrs[ptx][key][k]
                fillvalue = FILL_VALUE[ptx][key][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(ptx,key,k)
                if fillvalue:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
                else:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, compression='gzip')
                # create or attach dimensions for HDF5 variable
                if DIMENSIONS[ptx][key][k]:
                    # attach dimensions
                    for i,dim in enumerate(DIMENSIONS[ptx][key][k]):
                        if (key == 'cycle_stats'):
                            h5[ptx][key][k].dims[i].attach_scale(h5[ptx][dim])
                        else:
                            h5[ptx][key][k].dims[i].attach_scale(h5[ptx][key][dim])
                else:
                    # make dimension
                    h5[ptx][key][k].make_scale(k)
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[ptx][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Annual Land Ice Height'
    fileID.attrs['summary'] = ('The purpose of ATL11 is to provide an ICESat-2 '
        'satellite cycle summary of heights and height changes of land-based '
        'ice and will be provided as input to ATL15 and ATL16, gridded '
        'estimates of heights and height-changes.')
    fileID.attrs['description'] = ('Land ice parameters for each beam pair. '
        'All parameters are calculated for the same along-track increments '
        'for each beam pair and repeat.')
    date_created = datetime.datetime.today()
    fileID.attrs['date_created'] = date_created.isoformat()
    project = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = project
    platform = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = platform
    # add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    # add attributes for input ATL11 files
    fileID.attrs['input_files'] = os.path.basename(INPUT)
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for ptx in pairs:
        lon = IS2_atl11_corr[ptx]['longitude']
        lat = IS2_atl11_corr[ptx]['latitude']
        delta_time = IS2_atl11_corr[ptx]['delta_time']
        valid = np.nonzero(delta_time != FILL_VALUE[ptx]['delta_time'])
        # setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time[valid].min() if (delta_time[valid].min() < tmn) else tmn
        tmx = delta_time[valid].max() if (delta_time[valid].max() > tmx) else tmx
    # add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = ltmn
    fileID.attrs['geospatial_lat_max'] = ltmx
    fileID.attrs['geospatial_lon_min'] = lnmn
    fileID.attrs['geospatial_lon_max'] = lnmx
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['date_type'] = 'UTC'
    fileID.attrs['time_type'] = 'CCSDS UTC-A'
    # convert start and end time from ATLAS SDP seconds into GPS seconds
    atlas_sdp_gps_epoch=IS2_atl11_corr['ancillary_data']['atlas_sdp_gps_epoch']
    gps_seconds = atlas_sdp_gps_epoch + np.array([tmn,tmx])
    # calculate leap seconds
    leaps = icesat2_toolkit.time.count_leap_seconds(gps_seconds)
    # convert from seconds since 1980-01-06T00:00:00 to Julian days
    MJD = icesat2_toolkit.time.convert_delta_time(gps_seconds - leaps,
        epoch1=(1980,1,6,0,0,0), epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    YY,MM,DD,HH,MN,SS = icesat2_toolkit.time.convert_julian(MJD + 2400000.5,
        FORMAT='tuple')
    # add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = '{0:0.0f}'.format(tmx-tmn)
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create arguments parser
def arguments():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Calculates ice sheet grounding zones with ICESat-2
            ATL11 annual land ice height data
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL11 file to run')
    # directory with mask data
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=get_data_path('data'),
        help='Working data directory')
    # tide model to use
    model_choices = pyTMD.model.ocean_elevation()
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, default='CATS2008',
        choices=model_choices,
        help='Tide model to use in correction')
    ib_choices = ['ERA-Interim','ERA5','MERRA-2']
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str, choices=ib_choices,
        help='Reanalysis model to use in inverse-barometer correction')
    #-- run with ATL11 crossovers
    parser.add_argument('--crossovers','-C',
        default=False, action='store_true',
        help='Run ATL11 Crossovers')
    # create test plots
    parser.add_argument('--plot','-P',
        default=False, action='store_true',
        help='Create plots of flexural zone')
    # verbosity settings
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # run for each input ATL11 file
    for FILE in args.infile:
        calculate_GZ_ICESat2(args.directory, FILE, TIDE_MODEL=args.tide,
            REANALYSIS=args.reanalysis, CROSSOVERS=args.crossovers,
            PLOT=args.plot, VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()