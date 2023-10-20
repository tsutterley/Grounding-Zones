#!/usr/bin/env python
u"""
calculate_GZ_ICESat2_ATL06.py
Written by Tyler Sutterley (08/2023)

Calculates ice sheet grounding zones with ICESat-2 data following:
    Brunt et al., Annals of Glaciology, 51(55), 2010
        https://doi.org/10.3189/172756410791392790
    Fricker et al. Geophysical Research Letters, 33(15), 2006
        https://doi.org/10.1029/2006GL026907
    Fricker et al. Antarctic Science, 21(5), 2009
        https://doi.org/10.1017/S095410200999023X

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
    --mean-file X: Mean elevation file to remove from the height data
    -T X, --tide X: Tide model to use in correction
        CATS0201
        CATS2008
        TPXO9-atlas
        TPXO9-atlas-v2
        TPXO9-atlas-v3
        TPXO9-atlas-v4
        TPXO9-atlas-v5
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
    -S, --sea-level: Remove mean dynamic topography from heights
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
    io/ATL06.py: reads ICESat-2 ATL06 land ice data files
    convert_delta_time.py: converts from delta time into Julian and year-decimal
    time.py: utilities for calculating time operations
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        use time functions from timescale.time
    Updated 07/2023: using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 11/2022: verify coordinate reference system of shapefile
        output estimated grounding zone location for each beam to HDF5
    Updated 10/2022: use a defined mean height file for the baseline
    Updated 08/2022: use logging for verbose output of processing run
    Updated 07/2022: place shapely within try/except statement
    Updated 05/2022: use argparse descriptions within documentation
    Updated 03/2021: use utilities to set default path to shapefiles
        replaced numpy bool/int to prevent deprecation warnings
    Updated 01/2021: using argparse to set command line options
        using time module for conversion operations
    Updated 09/2019: using date functions paralleling public repository
    Updated 05/2019: check if beam exists in a try except else clause
    Updated 09/2017: use rcond=-1 in numpy least-squares algorithms
    Written 06/2017
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import datetime
import operator
import warnings
import itertools
import traceback
import collections
import numpy as np
import scipy.stats
import scipy.optimize
import grounding_zones as gz

# attempt imports
try:
    import fiona
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("fiona not available", ImportWarning)
try:
    import h5py
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("h5py not available", ImportWarning)
try:
    import icesat2_toolkit as is2tk
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("icesat2_toolkit not available", ImportWarning)
try:
    import matplotlib.pyplot as plt
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("matplotlib not available", ImportWarning)
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("pyproj not available", ImportWarning)
try:
    import shapely.geometry
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("shapely not available", ImportWarning)
try:
    import timescale
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("timescale not available", ImportWarning)

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
    input_shapefile = base_dir.joinpath(grounded_shapefile[HEM])
    shape = fiona.open(str(input_shapefile))
    # extract coordinate reference system
    if ('init' in shape.crs.keys()):
        epsg = pyproj.CRS(shape.crs['init']).to_epsg()
    else:
        epsg = pyproj.CRS(shape.crs).to_epsg()
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
    return (mline_obj, epsg)

# PURPOSE: attempt to read the mask variables
def read_grounding_zone_mask(mask_file, gtx):
    # check that mask file and variable exists
    for mask in ['ice_gz', 'mask']:
        try:
            # extract mask values to create grounding zone mask
            fileID = h5py.File(mask_file, mode='r')
            v1 = [gtx, 'land_ice_segments', 'subsetting', mask]
            # read buffered grounding zone mask
            ice_gz = fileID['/'.join(v1)][:].copy()
        except Exception as exc:
            logging.debug(traceback.format_exc())
            pass
        else:
            # close the HDF5 file and return the mask variable
            fileID.close()
            return ice_gz
    # raise value error
    raise KeyError(f'Cannot retrieve mask variable for {str(mask_file)}')

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

# PURPOSE: read ICESat-2 data from NSIDC or MPI_ICESat2_ATL03.py
# calculate mean elevation between all dates in file
# calculate inflexion point using elevation surface slopes
# use mean elevation to calculate elevation anomalies
# use anomalies to calculate inward and seaward limits of tidal flexure
def calculate_GZ_ICESat2(base_dir, INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    MEAN_FILE=None,
    TIDE_MODEL=None,
    REANALYSIS=None,
    SEA_LEVEL=False,
    PLOT=False,
    MODE=0o775):

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_'
        r'(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$', re.VERBOSE)
    PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = rx.findall(GRANULE).pop()
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRAN)

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = gz.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = gz.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()

    # read data from input_file
    IS2_atl06_mds,IS2_atl06_attrs,IS2_atl06_beams = \
        is2tk.io.ATL06.read_granule(INPUT_FILE,
                                    HISTOGRAM=False,
                                    QUALITY=False,
                                    ATTRIBUTES=True)

    # file format for auxiliary files
    file_format = '{0}_{1}_{2}_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.h5'
    plot_format='{0}_{1}_{2}_{3}_{4}{5}{6}{7}{8}{9}_{10}{11}{12}_{13}_{14}{15}.png'

    # grounded ice line string to determine if segment crosses coastline
    mline_obj, epsg = read_grounded_ice(base_dir, HEM, VARIABLES=[0])
    # projections for converting lat/lon to polar stereographic
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(epsg)
    # transformer object for converting projections
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # densities of seawater and ice
    rho_w = 1030.0
    rho_ice = 917.0

    # number of GPS seconds between the GPS epoch
    # and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = IS2_atl06_mds['ancillary_data']['atlas_sdp_gps_epoch']

    # copy variables for outputting to HDF5 file
    IS2_atl06_gz = {}
    IS2_atl06_fill = {}
    IS2_atl06_dims = {}
    IS2_atl06_gz_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl06_gz['ancillary_data'] = {}
    IS2_atl06_gz_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl06_gz['ancillary_data'][key] = IS2_atl06_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl06_gz_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][key].items():
            IS2_atl06_gz_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl06_beams):
        # number of segments
        v = IS2_atl06_mds[gtx]['land_ice_segments']
        attrs = IS2_atl06_attrs[gtx]['land_ice_segments']
        n_seg = len(v['segment_id'])
        # find valid segments for beam within grounding zone
        fv = attrs['h_li']['_FillValue']
        # land ice height
        h_li = np.ma.array(v['h_li'], fill_value=fv, mask=(v['h_li']==fv))

        # if creating a test plot
        if PLOT:
            fig1,ax1 = plt.subplots(num=1,figsize=(13,7))

        # flag that a valid grounding zone fit has been found
        valid_fit = False
        # outputs of grounding zone fit
        grounding_zone_data = {}
        grounding_zone_data['segment_id'] = []
        grounding_zone_data['latitude'] = []
        grounding_zone_data['longitude'] = []
        grounding_zone_data['delta_time'] = []
        # grounding_zone_data['tide_ocean'] = []
        grounding_zone_data['gz_sigma'] = []
        grounding_zone_data['e_mod'] = []
        grounding_zone_data['e_mod_sigma'] = []
        # grounding_zone_data['H_ice'] = []
        # grounding_zone_data['delta_h'] = []

        # grounding zone mask file
        a1 = (PRD,'GROUNDING_ZONE','MASK',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        f1 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a1))
        ice_gz = np.zeros((n_seg),dtype=bool)
        # check that mask file exists
        try:
            # extract mask values for mask flags to create grounding zone mask
            ice_gz[:] = read_grounding_zone_mask(f1, gtx)
        except Exception as exc:
            logging.debug(traceback.format_exc())
            continue

        # read mean elevation file (e.g. digital elevation model)
        dem_h = np.ma.zeros((n_seg), fill_value=fv)
        if MEAN_FILE:
            # read DEM HDF5 file
            try:
                fid2 = h5py.File(MEAN_FILE, mode='r')
                v2 = [gtx,'land_ice_segments','dem','dem_h']
                dem_h.data[:] = fid2['/'.join(v2)][:].copy()
                fv2 = fid2['/'.join(v2)].fillvalue
            except Exception as exc:
                logging.debug(traceback.format_exc())
                dem_h.mask = np.ones((n_seg),dtype=bool)
            else:
                dem_h.mask = (dem_h.data[:] == fv2)
                fid2.close()
        else:
            # use default DEM within ATL06
            dem_h.data[:] = v['dem']['dem_h'][:].copy()
            fv2 = attrs['dem']['dem_h']['_FillValue']
            dem_h.mask = (v['dem']['dem_h'][:] == fv2)

        # read tide model
        if TIDE_MODEL:
            # read tide model HDF5 file
            a3 = (PRD,TIDE_MODEL,'TIDES',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
            f3 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a3))
            tide_ocean = np.ma.zeros((n_seg),fill_value=fv)
            # check that tide file exists
            try:
                fid3 = h5py.File(f3,'r')
                v3 = [gtx,'land_ice_segments','geophysical','tide_ocean']
                tide_ocean.data[:] = fid3['/'.join(v3)][:].copy()
                fv3 = fid3['/'.join(v3)].fillvalue
            except Exception as exc:
                logging.debug(traceback.format_exc())
                tide_ocean.mask = np.ones((n_seg),dtype=bool)
            else:
                tide_ocean.mask = (tide_ocean.data[:] == fv3)
                fid3.close()
        else:
            # use default tide model
            tide_ocean = np.ma.array(v['geophysical']['tide_ocean'])
            fv3 = attrs['geophysical']['tide_ocean']['_FillValue']
            tide_ocean.mask = (tide_ocean.data[:] == fv3)

        # read inverse barometer correction
        if REANALYSIS:
            # read inverse barometer HDF5 file
            a4 = (PRD,REANALYSIS,'IB',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
            f4 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a4))
            IB = np.ma.zeros((n_seg),fill_value=fv)
            # check that inverse barometer exists
            try:
                fid4 = h5py.File(f4,'r')
                v4 = [gtx,'land_ice_segments','geophysical','ib']
                IB.data[:] = fid4['/'.join(v4)][:].copy()
                fv4 = fid4['/'.join(v4)].fillvalue
            except Exception as exc:
                logging.debug(traceback.format_exc())
                IB.mask = np.ones((n_seg),dtype=bool)
            else:
                IB.mask = (IB.data[:] == fv4)
                fid4.close()
        else:
            # use default dynamic atmospheric correction
            IB = np.ma.array(v['geophysical']['dac'])
            fv4 = attrs['geophysical']['dac']['_FillValue']
            IB.mask = (IB.data[:] == fv4)

        # mean dynamic topography
        mdt = np.ma.zeros((n_seg),fill_value=fv)
        if SEA_LEVEL:
            a5 = (PRD,'AVISO','SEA_LEVEL',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
            f5 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a5))
            # check that mean dynamic topography file exists
            try:
                fid5 = h5py.File(f5,'r')
                v5 = [gtx,'land_ice_segments','geophysical','h_mdt']
                mdt.data[:] = fid5['/'.join(v5)][:].copy()
                fv5 = fid5['/'.join(v5)].fillvalue
            except Exception as exc:
                logging.debug(traceback.format_exc())
                mdt.mask = np.ones((n_seg),dtype=bool)
                pass
            else:
                IB.mask = (IB.data[:] == fv5)
                fid5.close()
        else:
            # use no mean dynamic topography
            mdt.mask = np.zeros((n_seg),dtype=bool)

        # find valid points with GZ for both ATL06 and the interpolated DEM
        valid, = np.nonzero((~h_li.mask) & (~dem_h.mask) & ice_gz)

        # compress list (separate geosegs into sets of ranges)
        ice_gz_indices = compress_list(valid,10)
        for imin,imax in ice_gz_indices:
            # find valid indices within range
            i = sorted(set(np.arange(imin,imax+1)) & set(valid))
            # extract lat/lon and convert to polar stereographic
            X,Y = transformer.transform(v['longitude'][i],v['latitude'][i])
            # shapely LineString object for altimetry segment
            segment_line = shapely.geometry.LineString(np.c_[X, Y])
            # determine if line segment intersects previously known GZ
            if segment_line.intersects(mline_obj):
                # extract intersected point (find minimum distance)
                try:
                    xi,yi = mline_obj.intersection(segment_line).xy
                except:
                    continue
                else:
                    iint = np.argmin((Y-yi)**2 + (X-xi)**2)
                # horizontal eulerian distance from start of segment
                dist = np.sqrt((X-X[0])**2 + (Y-Y[0])**2)
                # land ice height for grounding zone
                h_gz = h_li.data[i]
                # mean land ice height from digital elevation model
                h_mean = dem_h.data[i]
                # geoid height
                geoid_h = v['dem']['geoid_h'][i]

                # ocean tide height for scaling model
                h_tide = np.ma.array(tide_ocean.data[i], fill_value=fv)
                h_tide.mask = tide_ocean.mask[i]
                # inverse-barometer response
                h_ib = np.ma.array(IB.data[i], fill_value=fv)
                h_ib.mask = IB.mask[i]

                # deflection from mean land ice height in grounding zone
                dh_gz = h_gz - h_mean
                # quasi-freeboard: WGS84 elevation - geoid height
                QFB = h_gz - (geoid_h + mdt[i])
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
                GZ = np.array([np.inf, np.inf])
                PGZ = np.array([np.inf, np.inf])
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
                            THICKNESS=w_thick, CONF=0.95, XOUT=i)
                    except Exception as exc:
                        logging.debug(traceback.format_exc())
                        pass
                    # copy grounding zone parameters to get best fit
                    if (GZ[1] < PGZ[1]):
                        PGZ = np.copy(GZ)
                        model_scale = np.copy(PA[0])
                        PEMODEL = np.copy(MODEL)
                    # use parameters if fit significance is within tolerance
                    if (GZ[1] < 400.0):
                        break
                # skip saving parameters if no valid solution was found
                if np.logical_not(np.isfinite(PGZ[0])):
                    continue
                # set valid beam fit flag
                valid_fit = True

                # linearly interpolate distance to grounding line
                GZseg = np.interp(PGZ[0],dist,v['segment_id'][i])
                GZlat = np.interp(PGZ[0],dist,v['latitude'][i])
                GZlon = np.interp(PGZ[0],dist,v['longitude'][i])
                GZtime = np.interp(PGZ[0],dist,v['delta_time'][i])
                # append outputs of grounding zone fit
                # save all outputs (not just within tolerance)
                grounding_zone_data['segment_id'].append(GZseg)
                grounding_zone_data['latitude'].append(GZlat)
                grounding_zone_data['longitude'].append(GZlon)
                grounding_zone_data['delta_time'].append(GZtime)
                # grounding_zone_data['tide_ocean'].append(PA)
                grounding_zone_data['gz_sigma'].append(PGZ[1])
                grounding_zone_data['e_mod'].append(PE[0]/1e9)
                grounding_zone_data['e_mod_sigma'].append(PE[1]/1e9)
                # grounding_zone_data['H_ice'].append(PT)
                # grounding_zone_data['delta_h'].append(PdH)

                # add to test plot
                if PLOT:
                    # plot height differences
                    l, = ax1.plot(v['segment_id'][i], dh_gz-PdH[0], '.', ms=1.5)
                    # plot grounding line location
                    ax1.axvline(GZseg, color=l.get_color(), ls='--', dashes=(8,4))

        # make final plot adjustments and save to file
        if PLOT and valid_fit:
            # adjust figure
            fig1.subplots_adjust(left=0.05,right=0.97,bottom=0.04,top=0.96,hspace=0.15)
            # create plot file of flexural zone
            args = (PRD,gtx,TIDE_MODEL,'GZ',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
            output_plot_file = OUTPUT_DIRECTORY.joinpath(plot_format.format(*args))
            # log output plot file
            logging.info(str(output_plot_file))
            fig1.savefig(output_plot_file, dpi=240, format='png',
                metadata={'Title':pathlib.Path(sys.argv[0]).name})
            output_plot_file.chmod(mode=MODE)
            # clear all figure axes
            plt.cla()
            plt.clf()

        # if no valid grounding zone fit has been found
        # skip saving variables and attributes for beam
        if not valid_fit:
            continue

        # output data dictionaries for beam
        IS2_atl06_gz[gtx] = dict(grounding_zone_data=collections.OrderedDict())
        IS2_atl06_fill[gtx] = dict(grounding_zone_data=collections.OrderedDict())
        IS2_atl06_dims[gtx] = dict(grounding_zone_data=collections.OrderedDict())
        IS2_atl06_gz_attrs[gtx] = dict(grounding_zone_data=collections.OrderedDict())

        # group attributes for beam
        IS2_atl06_gz_attrs[gtx]['Description'] = IS2_atl06_attrs[gtx]['Description']
        IS2_atl06_gz_attrs[gtx]['atlas_pce'] = IS2_atl06_attrs[gtx]['atlas_pce']
        IS2_atl06_gz_attrs[gtx]['atlas_beam_type'] = IS2_atl06_attrs[gtx]['atlas_beam_type']
        IS2_atl06_gz_attrs[gtx]['groundtrack_id'] = IS2_atl06_attrs[gtx]['groundtrack_id']
        IS2_atl06_gz_attrs[gtx]['atmosphere_profile'] = IS2_atl06_attrs[gtx]['atmosphere_profile']
        IS2_atl06_gz_attrs[gtx]['atlas_spot_number'] = IS2_atl06_attrs[gtx]['atlas_spot_number']
        IS2_atl06_gz_attrs[gtx]['sc_orientation'] = IS2_atl06_attrs[gtx]['sc_orientation']
        # group attributes for grounding zone variables
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['Description'] = ("The grounding_zone_data "
            "subgroup contains statistic data at grounding zone locations.")
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['data_rate'] = ("Data within this group are "
            "stored at the average segment rate.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl06_gz[gtx]['grounding_zone_data']['delta_time'] = np.copy(grounding_zone_data['delta_time'])
        IS2_atl06_fill[gtx]['grounding_zone_data']['delta_time'] = None
        IS2_atl06_dims[gtx]['grounding_zone_data']['delta_time'] = None
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['delta_time'] = collections.OrderedDict()
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['delta_time']['standard_name'] = "time"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['delta_time']['calendar'] = "standard"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['delta_time']['coordinates'] = \
            "segment_id latitude longitude"
        # segment ID
        IS2_atl06_gz[gtx]['grounding_zone_data']['segment_id'] = np.copy(grounding_zone_data['segment_id'])
        IS2_atl06_fill[gtx]['grounding_zone_data']['segment_id'] = None
        IS2_atl06_dims[gtx]['grounding_zone_data']['segment_id'] = ['delta_time']
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['segment_id'] = collections.OrderedDict()
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['segment_id']['units'] = "1"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['segment_id']['contentType'] = "referenceInformation"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['segment_id']['long_name'] = "Along-track segment ID number"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['segment_id']['description'] = ("A 7 digit number "
            "identifying the along-track geolocation segment number.  These are sequential, starting with "
            "1 for the first segment after an ascending equatorial crossing node. Equal to the segment_id for "
            "the second of the two 20m ATL03 segments included in the 40m ATL06 segment")
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['segment_id']['coordinates'] = \
            "delta_time latitude longitude"
        # latitude
        IS2_atl06_gz[gtx]['grounding_zone_data']['latitude'] = np.copy(grounding_zone_data['latitude'])
        IS2_atl06_fill[gtx]['grounding_zone_data']['latitude'] = None
        IS2_atl06_dims[gtx]['grounding_zone_data']['latitude'] = ['delta_time']
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude'] = collections.OrderedDict()
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude']['units'] = "degrees_north"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude']['long_name'] = "Latitude"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude']['standard_name'] = "latitude"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude']['description'] = ("Latitude of "
            "estimated grounding zone location")
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude']['valid_min'] = -90.0
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude']['valid_max'] = 90.0
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['latitude']['coordinates'] = \
            "segment_id delta_time longitude"
        # longitude
        IS2_atl06_gz[gtx]['grounding_zone_data']['longitude'] = np.copy(grounding_zone_data['longitude'])
        IS2_atl06_fill[gtx]['grounding_zone_data']['longitude'] = None
        IS2_atl06_dims[gtx]['grounding_zone_data']['longitude'] = ['delta_time']
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude'] = collections.OrderedDict()
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude']['units'] = "degrees_east"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude']['long_name'] = "Longitude"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude']['standard_name'] = "longitude"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude']['description'] = ("Longitude of "
            "estimated grounding zone location")
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude']['valid_min'] = -180.0
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude']['valid_max'] = 180.0
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['longitude']['coordinates'] = \
            "segment_id delta_time latitude"

        # uncertainty of the grounding zone
        IS2_atl06_gz[gtx]['grounding_zone_data']['gz_sigma'] = np.copy(grounding_zone_data['gz_sigma'])
        IS2_atl06_fill[gtx]['grounding_zone_data']['gz_sigma'] = 0.0
        IS2_atl06_dims[gtx]['grounding_zone_data']['gz_sigma'] = ['delta_time']
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['gz_sigma'] = collections.OrderedDict()
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['gz_sigma']['units'] = "meters"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['gz_sigma']['contentType'] = "physicalMeasurement"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['gz_sigma']['long_name'] = "grounding zone uncertainty"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['gz_sigma']['source'] = "ATL11"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['gz_sigma']['description'] = ("Uncertainty in grounding"
            "zone location derived by the physical elastic bending model")
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['gz_sigma']['coordinates'] = \
            "segment_id delta_time latitude longitude"
        # effective elastic modulus
        IS2_atl06_gz[gtx]['grounding_zone_data']['e_mod'] = np.copy(grounding_zone_data['e_mod'])
        IS2_atl06_fill[gtx]['grounding_zone_data']['e_mod'] = 0.0
        IS2_atl06_dims[gtx]['grounding_zone_data']['e_mod'] = ['delta_time']
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod'] = collections.OrderedDict()
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod']['units'] = "GPa"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod']['contentType'] = "physicalMeasurement"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod']['long_name'] = "Elastic modulus"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod']['source'] = "ATL11"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod']['description'] = ("Effective Elastic modulus "
            "of ice estimating using an elastic beam model")
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod']['coordinates'] = \
            "segment_id delta_time latitude longitude"
        # uncertainty of the elastic modulus
        IS2_atl06_gz[gtx]['grounding_zone_data']['e_mod_sigma'] = np.copy(grounding_zone_data['e_mod_sigma'])
        IS2_atl06_fill[gtx]['grounding_zone_data']['e_mod_sigma'] = 0.0
        IS2_atl06_dims[gtx]['grounding_zone_data']['e_mod_sigma'] = ['delta_time']
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod_sigma'] = collections.OrderedDict()
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod_sigma']['units'] = "GPa"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod_sigma']['contentType'] = "physicalMeasurement"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod_sigma']['long_name'] = "Elastic modulus uncertainty"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod_sigma']['source'] = "ATL11"
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod_sigma']['description'] = ("Uncertainty in the "
            "effective Elastic modulus of ice")
        IS2_atl06_gz_attrs[gtx]['grounding_zone_data']['e_mod_sigma']['coordinates'] = \
            "segment_id delta_time latitude longitude"

    # check that there are any valid beams in the dataset
    if bool([k for k in IS2_atl06_gz.keys() if bool(re.match(r'gt\d[lr]',k))]):
        # output HDF5 file for grounding zone locations
        fargs = (PRD,TIDE_MODEL,'GZ',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(file_format.format(*fargs))
        # print file information
        logging.info(f'\t{OUTPUT_FILE}')
        # write to output HDF5 file
        HDF5_ATL06_corr_write(IS2_atl06_gz, IS2_atl06_gz_attrs,
            FILENAME=OUTPUT_FILE,
            INPUT=GRANULE,
            FILL_VALUE=IS2_atl06_fill,
            DIMENSIONS=IS2_atl06_dims,
            CLOBBER=True)
        # change the permissions mode
        OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the grounding zone data for ICESat-2 data to HDF5
def HDF5_ATL06_corr_write(IS2_atl06_corr, IS2_atl06_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=True):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    fileID = h5py.File(FILENAME, clobber)

    # create HDF5 records
    h5 = {}

    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl06_corr['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl06_corr.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl06_attrs[gtx][att_name]
        # create grounding_zone_data group
        fileID[gtx].create_group('grounding_zone_data')
        h5[gtx] = dict(grounding_zone_data={})
        for att_name in ['Description','data_rate']:
            att_val = IS2_atl06_attrs[gtx]['grounding_zone_data'][att_name]
            fileID[gtx]['grounding_zone_data'].attrs[att_name] = att_val

        # segment_id, geolocation, time and height variables
        for k,v in IS2_atl06_corr[gtx]['grounding_zone_data'].items():
            # values and attributes
            attrs = IS2_atl06_attrs[gtx]['grounding_zone_data'][k]
            fillvalue = FILL_VALUE[gtx]['grounding_zone_data'][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}'.format(gtx,'grounding_zone_data',k)
            if fillvalue:
                h5[gtx]['grounding_zone_data'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                    compression='gzip')
            else:
                h5[gtx]['grounding_zone_data'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[gtx]['grounding_zone_data'][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['grounding_zone_data'][k]):
                    h5[gtx]['grounding_zone_data'][k].dims[i].attach_scale(
                        h5[gtx]['grounding_zone_data'][dim])
            else:
                # make dimension
                h5[gtx]['grounding_zone_data'][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['grounding_zone_data'][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Land Ice Grounding Zones'
    fileID.attrs['summary'] = ('Grounding zone data for ice-sheets segments '
        'estimated using an elastic beam fit.')
    fileID.attrs['description'] = ('Land ice parameters for each beam.  All '
        'grounding zone are calculated for each beam.')
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
    # add attributes for input ATL06 file
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl06_corr[gtx]['grounding_zone_data']['longitude']
        lat = IS2_atl06_corr[gtx]['grounding_zone_data']['latitude']
        delta_time = IS2_atl06_corr[gtx]['grounding_zone_data']['delta_time']
        # setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time.min() if (delta_time.min() < tmn) else tmn
        tmx = delta_time.max() if (delta_time.max() > tmx) else tmx
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
    # convert start and end time from ATLAS SDP seconds into timescale
    ts = timescale.time.Timescale().from_deltatime(np.array([tmn,tmx]),
        epoch=timescale.time._atlas_sdp_epoch, standard='GPS')
    dt = np.datetime_as_string(ts.to_datetime(), unit='s')
    # add attributes with measurement date start, end and duration
    fileID.attrs['time_coverage_start'] = str(dt[0])
    fileID.attrs['time_coverage_end'] = str(dt[1])
    fileID.attrs['time_coverage_duration'] = f'{tmx-tmn:0.0f}'
    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates ice sheet grounding zones with ICESat-2
            ATL06 along-track land ice height data
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL06 file to run')
    # directory with mask data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=gz.utilities.get_data_path('data'),
        help='Working data directory')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # mean file to remove
    parser.add_argument('--mean-file',
        type=pathlib.Path,
        help='Mean elevation file to remove from the height data')
    # tide model to use
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, default='CATS2008',
        help='Tide model to use in correction')
    # dynamic atmospheric correction
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str,
        help='Reanalysis model to use in inverse-barometer correction')
    # mean dynamic topography
    parser.add_argument('--sea-level','-S',
        default=False, action='store_true',
        help='Remove mean dynamic topography from heights')
    # create test plots
    parser.add_argument('--plot','-P',
        default=False, action='store_true',
        help='Create plots of flexural zone')
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

    # create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # run for each input ATL06 file
    for FILE in args.infile:
        calculate_GZ_ICESat2(args.directory, FILE,
            MEAN_FILE=args.mean_file, TIDE_MODEL=args.tide,
            REANALYSIS=args.reanalysis, SEA_LEVEL=args.sea_level,
            PLOT=args.plot, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()