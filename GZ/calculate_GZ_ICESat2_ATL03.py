#!/usr/bin/env python
u"""
calculate_GZ_ICESat2_ATL03.py
Written by Tyler Sutterley (08/2022)
Calculates ice sheet grounding zones with ICESat-2 data following:
    Brunt et al., Annals of Glaciology, 51(55), 2010
        https://doi.org/10.3189/172756410791392790
    Fricker et al. Geophysical Research Letters, 33(15), 2006
        https://doi.org/10.1029/2006GL026907
    Fricker et al. Antarctic Science, 21(5), 2009
        https://doi.org/10.1017/S095410200999023X

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python (Spatial algorithms and data structures)
        https://docs.scipy.org/doc/
        https://docs.scipy.org/doc/scipy/reference/spatial.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    fiona: Python wrapper for vector data access functions from the OGR library
        https://fiona.readthedocs.io/en/latest/manual.html
    shapely: PostGIS-ish operations outside a database context for Python
        http://toblerity.org/shapely/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    read_ICESat2_ATL03.py: reads ICESat-2 ATL03 and ATL09 data files
    time.py: utilities for calculating time operations
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 08/2022: use logging for verbose output of processing run
    Updated 07/2022: place shapely within try/except statement
    Updated 05/2022: use argparse descriptions within documentation
    Updated 03/2021: use utilities to set default path to shapefiles
        replaced numpy bool/int to prevent deprecation warnings
    Updated 01/2021: using argparse to set command line options
        using time module for conversion operations
    Updated 09/2019: using date functions paralleling public repository
    Updated 09/2017: use rcond=-1 in numpy least-squares algorithms
    Written 06/2017
"""
from __future__ import print_function

import os
import re
import h5py
import pyproj
import logging
import argparse
import operator
import warnings
import itertools
import numpy as np
import scipy.stats
import scipy.optimize
import icesat2_toolkit.time
from grounding_zones.utilities import get_data_path
from icesat2_toolkit.read_ICESat2_ATL03 import read_HDF5_ATL03_main, \
    read_HDF5_ATL03_beam
#-- attempt imports
try:
    import fiona
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("fiona not available")
try:
    import shapely.geometry
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("shapely not available")
#-- ignore warnings
warnings.filterwarnings("ignore")

#-- grounded ice shapefiles
grounded_shapefile = {}
grounded_shapefile['N'] = 'grn_ice_sheet_peripheral_glaciers.shp'
grounded_shapefile['S'] = 'ant_ice_sheet_islands_v2.shp'
#-- description and reference for each grounded ice file
grounded_description = {}
grounded_description['N'] = 'Greenland Mapping Project (GIMP) Ice & Ocean Mask'
grounded_description['S'] = ('MEaSUREs Antarctic Boundaries for IPY 2007-2009 '
    'from Satellite_Radar, Version 2')
grounded_reference = {}
grounded_reference['N'] = 'https://doi.org/10.5194/tc-8-1509-2014'
grounded_reference['S'] = 'https://doi.org/10.5067/IKBWW4RYHF1Q'

#-- PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        projection_flag = 'S'
    elif GRANULE in ('03','04','05'):
        projection_flag = 'N'
    return projection_flag

#-- PURPOSE: find if segment crosses previously-known grounding line position
def read_grounded_ice(base_dir, HEM, VARIABLES=[0]):
    #-- reading grounded ice shapefile
    shape = fiona.open(os.path.join(base_dir,grounded_shapefile[HEM]))
    epsg = shape.crs['init']
    #-- reduce to variables of interest if specified
    shape_entities = [f for f in shape.values() if int(f['id']) in VARIABLES]
    #-- create list of polygons
    polygons = []
    #-- extract the entities and assign by tile name
    for i,ent in enumerate(shape_entities):
        #-- extract coordinates for entity
        poly_obj = shapely.geometry.Polygon(ent['geometry']['coordinates'])
        #-- Valid Polygon cannot have overlapping exterior or interior rings
        if (not poly_obj.is_valid):
            poly_obj = poly_obj.buffer(0)
        polygons.append(poly_obj)
    #-- create shapely multipolygon object
    mpoly_obj = shapely.geometry.MultiPolygon(polygons)
    #-- close the shapefile
    shape.close()
    #-- return the polygon object for the ice sheet
    return (mpoly_obj,grounded_shapefile[HEM],epsg)

#-- PURPOSE: compress complete list of valid indices into a set of ranges
def compress_list(i,n):
    for a,b in itertools.groupby(enumerate(i), lambda v: ((v[1]-v[0])//n)*n):
        group = list(map(operator.itemgetter(1),b))
        yield (group[0], group[-1])

#-- Derivation of Sharp Breakpoint Piecewise Regression:
#-- http://www.esajournals.org/doi/abs/10.1890/02-0472
#-- y = beta_0 + beta_1*t + e (for x <= alpha)
#-- y = beta_0 + beta_1*t + beta_2*(t-alpha) + e (for x > alpha)
def piecewise_fit(x, y, STEP=1, CONF=0.95):
    #-- regrid x and y to STEP
    XI = x[::STEP]
    YI = y[::STEP]
    #-- Creating Design matrix based on chosen input fit_type parameters:
    nmax = len(XI)
    P_x0 = np.ones((nmax))#-- Constant Term
    P_x1a = XI[0:nmax]#-- Linear Term 1
    #-- Calculating the number parameters to search
    n_param = (nmax**2 - nmax)//2
    #-- R^2 and Log-Likelihood
    rsquare_array = np.zeros((n_param))
    loglik_array = np.zeros((n_param))
    #-- output cutoff and fit parameters
    cutoff_array = np.zeros((n_param,2),dtype=int)
    beta_matrix = np.zeros((n_param,4))
    #-- counter variable
    c = 0
    #-- SStotal = sum((Y-mean(Y))^2)
    SStotal = np.dot(np.transpose(YI - np.mean(YI)),(YI - np.mean(YI)))
    #-- uniform distribution over entire range
    for n in range(0,nmax):
        #-- Linear Term 2 (= change from linear term1: trend2 = beta1+beta2)
        P_x1b = np.zeros((nmax))
        P_x1b[n:nmax] = XI[n:nmax] - XI[n]
        for nn in range(n+1,nmax):
            #-- Linear Term 3 (= change from linear term2)
            P_x1c = np.zeros((nmax))
            P_x1c[nn:nmax] = XI[nn:nmax] - XI[nn]
            DMAT = np.transpose([P_x0, P_x1a, P_x1b, P_x1c])
            #-- Calculating Least-Squares Coefficients
            #-- Least-Squares fitting (the [0] denotes coefficients output)
            beta_mat = np.linalg.lstsq(DMAT,YI,rcond=-1)[0]
            #-- number of terms in least-squares solution
            n_terms = len(beta_mat)
            #-- nu = Degrees of Freedom
            #-- number of measurements-number of parameters
            nu = nmax - n_terms
            #-- residual of data-model
            residual = YI - np.dot(DMAT,beta_mat)
            #-- CALCULATING R_SQUARE VALUES
            #-- SSerror = sum((Y-X*B)^2)
            SSerror = np.dot(np.transpose(residual),residual)
            #-- R^2 term = 1- SSerror/SStotal
            rsquare_array[c] = 1 - (SSerror/SStotal)
            #-- Log-Likelihood
            loglik_array[c] = 0.5*(-nmax*(np.log(2.0 * np.pi) + 1.0 - \
                np.log(nmax) + np.log(np.sum(residual**2))))
            #-- save cutoffs and beta matrix
            cutoff_array[c,:] = [n,nn]
            beta_matrix[c,:] = beta_mat
            #-- add 1 to counter
            c += 1

    #-- find where Log-Likelihood is maximum
    ind, = np.nonzero(loglik_array == loglik_array.max())
    n,nn = cutoff_array[ind,:][0]
    #-- create matrix of likelihoods
    likelihood = np.zeros((nmax,nmax))
    likelihood[:,:] = np.nan
    likelihood[cutoff_array[:,0],cutoff_array[:,1]] = np.exp(loglik_array) / \
        np.sum(np.exp(loglik_array))
    #-- probability distribution functions of each cutoff
    PDF1 = np.zeros((nmax))
    PDF2 = np.zeros((nmax))
    for i in range(nmax):
        #-- PDF for cutoff 1 for all cutoff 2
        PDF1[i] = np.nansum(likelihood[i,:])
        #-- PDF for cutoff 2 for all cutoff 1
        PDF2[i] = np.nansum(likelihood[:,i])
    #-- calculate confidence intervals
    # CI1 = conf_interval(XI, PDF1/np.sum(PDF1), CONF)
    CI1 = 5e3
    CMN1,CMX1 = (XI[n]-CI1,XI[nn]+CI1)
    # CI2 = conf_interval(XI, PDF2/np.sum(PDF2), CONF)
    CI2 = 5e3
    CMN2,CMX2 = (XI[nn]-CI2,XI[nn]+CI2)

    #-- calculate model using best fit coefficients
    P_x0 = np.ones_like(x)
    P_x1a = np.copy(x)
    P_x1b = np.zeros_like(x)
    P_x1c = np.zeros_like(x)
    P_x1b[n*STEP:] = x[n*STEP:] - XI[n]
    P_x1c[nn*STEP:] = x[nn*STEP:] - XI[nn]
    DMAT = np.transpose([P_x0, P_x1a, P_x1b, P_x1c])
    beta_mat, = beta_matrix[ind,:]
    MODEL = np.dot(DMAT,beta_mat)
    #-- return the cutoff parameters, their confidence interval and the model
    return ([XI[n],CMN1,CMX1], [XI[nn],CMN2,CMX2], MODEL)

#-- PURPOSE: run a physical elastic bending model with Levenberg-Marquardt
#-- D. G. Vaughan, Journal of Geophysical Research Solid Earth, 1995
#-- A. M. Smith, Journal of Glaciology, 1991
def physical_elastic_model(XI,YI,GZ=[0,0,0],METHOD='trf',ORIENTATION=False,
    THICKNESS=None,CONF=0.95):
    #-- reorient input parameters to go from land ice to floating
    if ORIENTATION:
        Xm1 = XI[-1]
        GZ = Xm1 - GZ
        GZ[1:] = GZ[:0:-1]
        XI = Xm1 - XI[::-1]
        YI = YI[::-1]
    #-- calculate thickness mean, min and max
    if THICKNESS is not None:
        #-- only use positive thickness values
        #-- ocean points could be negative with tides
        ii, = np.nonzero(THICKNESS > 0.0)
        MTH = np.mean(THICKNESS[ii])
        MNTH = np.min(THICKNESS[ii])
        MXTH = np.max(THICKNESS[ii])
    else:
        MTH = 1000.0
        MNTH = 100.0
        MXTH = 1900.0
    #-- elastic model parameters
    #-- G0: location of grounding line
    #-- A0: tidal amplitude (values from Padman 2002)
    #-- E0: Effective Elastic modulus of ice [Pa]
    #-- T0: ice thickness of ice shelf [m]
    #-- dH0: mean height change (thinning/thickening)
    p0 = [GZ[0], 1.2, 1e9, MTH, 0.0]
    #-- tuple for parameter bounds (lower and upper)
    #-- G0: 95% confidence interval of initial fit
    #-- A0: greater than +/- 2.4m value from Padman (2002)
    #-- E0: Range from Table 1 of Vaughan (1995)
    #-- T0: Range of ice thicknesses from Chuter (2015)
    #-- dH0: mean height change +/- 10 m/yr
    bounds = ([GZ[1], -3.0, 8.3e8, MNTH, -10],[GZ[2], 3.0, 1e10, MXTH, 10])
    #-- optimized curve fit with Levenberg-Marquardt algorithm
    popt,pcov = scipy.optimize.curve_fit(elasticmodel, XI, YI,
        p0=p0, bounds=bounds, method=METHOD)
    MODEL = elasticmodel(XI, *popt)
    #-- elasticmodel function outputs and 1 standard devation uncertainties
    GZ = np.zeros((2))
    A = np.zeros((2))
    E = np.zeros((2))
    T = np.zeros((2))
    dH = np.zeros((2))
    GZ[0],A[0],E[0],T[0],dH[0] = popt[:]
    #-- Error analysis
    #-- nu = Degrees of Freedom = number of measurements-number of parameters
    nu = len(XI) - len(p0)
    #-- Setting the confidence interval of the output error
    alpha = 1.0 - CONF
    #-- Student T-Distribution with D.O.F. nu
    #-- t.ppf parallels tinv in matlab
    tstar = scipy.stats.t.ppf(1.0-(alpha/2.0),nu)
    #-- error for each coefficient = t(nu,1-alpha/2)*standard error
    perr = np.sqrt(np.diag(pcov))
    GZ[1],A[1],E[1],T[1],dH[1] = tstar*perr[:]
    #-- reverse the reorientation
    if ORIENTATION:
        GZ[0] = Xm1 - GZ[0]
        MODEL = MODEL[::-1]
    return (GZ,A,E,T,dH,MODEL)

#-- PURPOSE: create physical elastic bending model with a mean height change
def elasticmodel(x, GZ, A, E, T, dH):
    #-- density of water [kg/m^3]
    rho_w = 1030.0
    #-- gravitational constant [m/s^2]
    g = 9.806
    #-- Poisson's ratio of ice
    nu = 0.3
    #-- structural rigidity of ice
    D = (E*T**3)/(12.0*(1.0-nu**2))
    #-- beta elastic damping parameter
    b = (0.25*rho_w*g/D)**0.25
    #-- distance of points from grounding line (R0 = 0 at grounding line)
    R0 = (x[x >= GZ] - GZ)
    #-- deflection of ice beyond the grounding line (elastic)
    eta = np.zeros_like(x)
    eta[x >= GZ] = A*(1.0-np.exp(-b*R0)*(np.cos(b*R0) + np.sin(b*R0)))
    #-- model = large scale height change + tidal deflection
    return (dH + eta)

#-- PURPOSE: calculate the confidence interval in the retrieval
def conf_interval(x,f,p):
    #-- sorting probability distribution from smallest probability to largest
    ii = np.argsort(f)
    #-- compute the sorted cumulative probability distribution
    cdf = np.cumsum(f[ii])
    #-- linearly interpolate to confidence interval
    J = np.interp(p, cdf, x[ii])
    #-- position with maximum probability
    K = x[ii[-1]]
    return np.abs(K-J)

#-- PURPOSE: read ICESat-2 reference photon event data from NSIDC
#-- calculate mean elevation between all dates in file
#-- calculate inflexion point using elevation surface slopes
#-- use mean elevation to calculate elevation anomalies
#-- use anomalies to calculate inward and seaward limits of tidal flexure
def calculate_GZ_ICESat2(base_dir, FILE, MODE=0o775):
    #-- print file information
    logging.info(os.path.basename(FILE))

    #-- read data from input_file
    IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams = read_HDF5_ATL03_main(FILE,
        ATTRIBUTES=True)
    DIRECTORY = os.path.dirname(FILE)
    #-- extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_'
        r'(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$',re.VERBOSE)
    PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = rx.findall(FILE).pop()
    #-- set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRAN)
    #-- digital elevation model for each region
    DEM_MODEL = dict(N='ArcticDEM', S='REMA')

    #-- file format for auxiliary files
    file_format='{0}_{1}_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
    #-- grounding zone mask file
    args = (PRD,'GROUNDING_ZONE_MASK',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
    #-- extract mask values for mask flags to create grounding zone mask
    fid1 = h5py.File(os.path.join(DIRECTORY,file_format.format(*args)), 'r')
    #-- input digital elevation model file (ArcticDEM or REMA)
    args = (PRD,DEM_MODEL[HEM],YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
    fid2 = h5py.File(os.path.join(DIRECTORY,file_format.format(*args)), 'r')
    # #-- input sea level for mean dynamic topography
    # args = (PRD,'AVISO_SEA_LEVEL',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
    # fid3 = h5py.File(os.path.join(DIRECTORY,file_format.format(*args)), 'r')

    #-- grounded ice line string to determine if segment crosses coastline
    mpoly_obj,input_file,epsg = read_grounded_ice(base_dir, HEM)
    #-- projections for converting lat/lon to polar stereographic
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    crs2 = pyproj.CRS.from_string(epsg)
    #-- transformer object for converting projections
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    #-- densities of seawater and ice
    rho_w = 1030.0
    rho_ice = 917.0

    #-- number of GPS seconds between the GPS epoch
    #-- and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = IS2_atl03_mds['ancillary_data']['atlas_sdp_gps_epoch']

    #-- copy variables for outputting to HDF5 file
    IS2_atl03_gz = {}
    IS2_atl03_fill = {}
    IS2_atl03_dims = {}
    IS2_atl03_gz_attrs = {}
    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    #-- Add this value to delta time parameters to compute full gps_seconds
    IS2_atl03_gz['ancillary_data'] = {}
    IS2_atl03_gz_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        #-- get each HDF5 variable
        IS2_atl03_gz['ancillary_data'][key] = IS2_atl03_mds['ancillary_data'][key]
        #-- Getting attributes of group and included variables
        IS2_atl03_gz_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][key].items():
            IS2_atl03_gz_attrs['ancillary_data'][key][att_name] = att_val

    #-- for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        #-- data and attributes for beam gtx
        val,attrs = read_HDF5_ATL03_beam(FILE,gtx,ATTRIBUTES=True,VERBOSE=False)
        #-- first photon in the segment (convert to 0-based indexing)
        Segment_Index_begin = val['geolocation']['ph_index_beg'] - 1
        #-- number of photon events in the segment
        Segment_PE_count = val['geolocation']['segment_ph_cnt']

        #-- number of photon events
        n_pe, = val['heights']['h_ph'].shape
        #-- invalid value
        fv = val['geolocation']['sigma_h'].fillvalue

        #-- check confidence level associated with each photon event
        #-- -1: Events not associated with a specific surface type
        #--  0: noise
        #--  1: buffer but algorithm classifies as background
        #--  2: low
        #--  3: medium
        #--  4: high
        #-- Surface types for signal classification confidence
        #-- 0=Land; 1=Ocean; 2=SeaIce; 3=LandIce; 4=InlandWater
        ice_sig_conf = np.copy(val['heights']['signal_conf_ph'][:,3])
        ice_sig_low_count = np.count_nonzero(ice_sig_conf > 1)

        #-- read buffered grounding zone mask
        ice_gz = fid1[gtx]['subsetting']['ice_gz'][:]
        B = fid1[gtx]['subsetting']['ice_gz'].attrs['source']

        #-- photon event height
        h_ph = np.ma.array(val['heights']['h_ph'], fill_value=fv,
            mask=(ice_sig_conf<=1))
        #-- digital elevation model elevation
        dem_h = np.ma.array(fid2[gtx]['heights']['dem_h'][:],
            mask=(fid2[gtx]['heights']['dem_h'][:]==fv), fill_value=fv)
        # #-- mean dynamic topography with invalid values set to 0
        # h_mdt = fid3[gtx]['geophys_corr']['h_mdt'][:]
        # h_mdt[h_mdt == fv] = 0.0

        #-- ocean tide
        tide = np.zeros_like(val['heights']['h_ph'],dtype=int)
        #-- dynamic atmospheric correction
        dac = np.zeros_like(val['heights']['h_ph'],dtype=int)
        #-- geoid height
        geoid = np.zeros_like(val['heights']['h_ph'],dtype=int)
        for j,idx in enumerate(Segment_Index_begin):
            #-- number of photons in segment
            cnt = Segment_PE_count[j]
            #-- get ocean tide for each photon event
            tide[idx:idx+cnt] = np.full((cnt),val['geophys_corr']['tide_ocean'][j])
            #-- get dynamic atmospheric correction for each photon event
            dac[idx:idx+cnt] = np.full((cnt),val['geophys_corr']['dac'][j])
            #-- get geoid height for each photon event
            geoid[idx:idx+cnt] = np.full((cnt),val['geophys_corr']['geoid'][j])

        #-- find valid points with GZ for both ATL03 and the interpolated DEM
        valid, = np.nonzero((~h_ph.mask) & (~dem_h.mask) & ice_gz)

        #-- compress list (separate geosegs into sets of ranges)
        ice_gz_indices = compress_list(valid,10)
        for imin,imax in ice_gz_indices:
            #-- find valid indices within range
            i = sorted(set(np.arange(imin,imax+1)) & set(valid))
            #-- convert time from ATLAS SDP to days relative into Julian days
            gps_seconds = atlas_sdp_gps_epoch + val['delta_time'][i]
            time_leaps = icesat2_toolkit.time.count_leap_seconds(gps_seconds)
            #-- convert from seconds since 1980-01-06T00:00:00 to Julian days
            time_julian = 2400000.5 + icesat2_toolkit.time.convert_delta_time(
                gps_seconds - time_leaps, epoch1=(1980,1,6,0,0,0),
                epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
            #-- convert to calendar date with convert_julian.py
            cal_date = icesat2_toolkit.time.convert_julian(time_julian)
            #-- extract lat/lon and convert to polar stereographic
            X,Y = transformer.transform(val['lon_ph'][i],val['lat_ph'][i])
            #-- shapely LineString object for altimetry segment
            segment_line = shapely.geometry.LineString(list(zip(X, Y)))
            #-- determine if line segment intersects previously known GZ
            if segment_line.intersects(mpoly_obj):
                #-- horizontal eulerian distance from start of segment
                dist = np.sqrt((X-X[0])**2 + (Y-Y[0])**2)
                #-- land ice height for grounding zone
                h_gz = h_ph.data[i]
                #-- mean land ice height from digital elevation model
                h_mean = dem_h[i]

                #-- deflection from mean height in grounding zone
                dh_gz = h_gz + tide[i] - h_mean # + dac[i]
                #-- quasi-freeboard: WGS84 elevation - geoid height
                QFB = h_gz - geoid[i]
                #-- ice thickness from quasi-freeboard and densities
                w_thick = QFB*rho_w/(rho_w-rho_ice)
                #-- fit with a hard piecewise model to get rough estimate of GZ
                C1,C2,PWMODEL = piecewise_fit(dist, dh_gz, STEP=5, CONF=0.95)
                #-- distance from estimated grounding line (0 = grounding line)
                d = (dist - C1[0]).astype(int)
                #-- determine if spacecraft is approaching coastline
                sco = True if np.mean(h_gz[d<0]) < np.mean(h_gz[d>0]) else False
                #-- fit physical elastic model
                PGZ,PA,PE,PT,PdH,PEMODEL = physical_elastic_model(dist, dh_gz,
                    GZ=C1, ORIENTATION=sco, THICKNESS=w_thick, CONF=0.95)
                #-- linearly interpolate distance to grounding line
                XGZ = np.interp(PGZ[0],dist,X)
                YGZ = np.interp(PGZ[0],dist,X)
                print(XGZ, YGZ, PGZ[0], PGZ[1], PT)

    #-- close the auxiliary files
    fid1.close()
    fid2.close()
    # fid3.close()

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates ice sheet grounding zones with ICESat-2
            ATL03 geolocated photon height data
            """
    )
    #-- command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL03 file to run')
    #-- directory with mask data
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=get_data_path('data'),
        help='Working data directory')
    #-- verbosity settings
    #-- verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    #-- permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    #-- return the parser
    return parser

#-- This is the main part of the program that calls the individual functions
def main():
    #-- Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    #-- create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- run for each input ATL03 file
    for FILE in args.infile:
        calculate_GZ_ICESat2(args.directory, FILE, MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
