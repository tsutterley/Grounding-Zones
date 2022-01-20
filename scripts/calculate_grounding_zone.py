#!/usr/bin/env python
u"""
calculate_grounding_zone.py
Written by Tyler Sutterley (01/2021)
Calculates ice sheet grounding zones following:
    Brunt et al., Annals of Glaciology, 51(55), 2010
        https://doi.org/10.3189/172756410791392790
    Fricker et al. Geophysical Research Letters, 33(15), 2006
        https://doi.org/10.1029/2006GL026907
    Fricker et al. Antarctic Science, 21(5), 2009
        https://doi.org/10.1017/S095410200999023X

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    --model X: Set the digital elevation model to run
        REMA
        ArcticDEM
        GIMP
    -F X, --format X: input and output data format
        csv (default)
        netCDF4
        HDF5
    -v X, --variables X: variable names of data in csv, HDF5 or netCDF4 file
        for csv files: the order of the columns within the file
        for HDF5 and netCDF4 files: time, y, x and data variable names
    -H X, --header X: number of header lines for csv files
    -P X, --projection X: spatial projection as EPSG code or PROJ4 string
        4326: latitude and longitude coordinates on WGS84 reference ellipsoid
    -V, --verbose: Verbose output of processing run
    -M X, --mode X: Permission mode of output file

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
    time.py: utilities for calculating time operations
    spatial.py: utilities for reading and writing spatial data
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 01/2021: using argparse to set command line options
        use pyTMD spatial module for reading and writing data
    Updated 09/2019: using date functions paralleling public repository
    Updated 05/2019: check if beam exists in a try except else clause
    Updated 09/2017: use rcond=-1 in numpy least-squares algorithms
    Written 06/2017
"""
from __future__ import print_function

import sys
import os
import re
import h5py
import fiona
import pyproj
import argparse
import operator
import itertools
import numpy as np
import scipy.stats
import scipy.optimize
import shapely.geometry
import matplotlib.pyplot as plt
import pyTMD.spatial

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

#-- PURPOSE: reading the number of file lines removing commented lines
def file_length(input_file):
    #-- read the input file, split at lines and remove all commented lines
    with open(input_file,'r') as f:
        i = [i for i in f.readlines() if re.match(r'^(?!\#|\n)',i)]
    #-- return the number of lines
    return len(i)

#-- PURPOSE: find if segment crosses previously-known grounding line position
def read_grounded_ice(base_dir, HEM, VARIABLES=[0]):
    #-- reading grounded ice shapefile
    shape = fiona.open(os.path.join(base_dir,grounded_shapefile[HEM]))
    epsg = shape.crs['init']
    #-- reduce to variables of interest if specified
    shape_entities = [f for f in shape.values() if np.int(f['id']) in VARIABLES]
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
    cutoff_array = np.zeros((n_param,2),dtype=np.int)
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
            #--- Least-Squares fitting (the [0] denotes coefficients output)
            beta_mat = np.linalg.lstsq(DMAT,YI,rcond=-1)[0]
            #--- number of terms in least-squares solution
            n_terms = len(beta_mat)
            #--- nu = Degrees of Freedom
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
#-- density of water [kg/m^3]
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
    # #-- compute the sorted cumulative probability distribution
    cdf = np.cumsum(f[ii])
    #-- find the min and max interval that contains the probability
    jj = np.max(np.nonzero(cdf < p))
    kk = np.min(np.nonzero(cdf >= p))
    #-- linearly interpolate to confidence interval
    J = x[ii[jj]] + (p - cdf[jj])/(cdf[kk] - cdf[jj])*(x[ii[kk]] - x[ii[jj]])
    K = x[ii[-1]]
    return np.abs(K-J)

#-- PURPOSE: read csv file of lat,lon coordinates
#-- calculate inflexion point using elevation surface slopes
#-- use mean elevation to calculate elevation anomalies
#-- use anomalies to calculate inward and seaward limits of tidal flexure
def calculate_grounding_zone(base_dir, input_file, output_file,
    DEM_MODEL=None, FORMAT='csv', VARIABLES=['time','lat','lon','data'],
    HEADER=0, PROJECTION='4326', VERBOSE=False, MODE=0o775):
    #-- get directory from input_file
    DIRECTORY = os.path.dirname(input_file)
    #-- set hemisphere flag based on digital elevation model
    hem_flag = dict(ArcticDEM='N', GIMP='N', REMA='S')
    HEM = hem_flag[DEM_MODEL]

    #-- read input file to extract time, spatial coordinates and data
    #-- read dem file from MPI_interpolate_DEM.py
    #-- output text file
    fileBasename,fileExtension = os.path.splitext(input_file)
    f2 = '{0}_{1}{2}'.format(fileBasename,DEM_MODEL,fileExtension)
    if (FORMAT == 'csv'):
        d1 = pyTMD.spatial.from_ascii(input_file, columns=VARIABLES,
            header=HEADER, verbose=VERBOSE)
        v2 = [VARIABLES[0], VARIABLES[1], VARIABLES[2], 'dem_h']
        d2 = pyTMD.spatial.from_ascii(f2, columns=v2,
            header=0, verbose=VERBOSE)
    elif (FORMAT == 'netCDF4'):
        d1 = pyTMD.spatial.from_netCDF4(input_file,
            xname=VARIABLES[2], yname=VARIABLES[1],
            timename=VARIABLES[0], varname=VARIABLES[3],
            verbose=VERBOSE)
        d2 = pyTMD.spatial.from_netCDF4(f2,
            xname=VARIABLES[2], yname=VARIABLES[1],
            timename=VARIABLES[0], varname='dem_h',
            verbose=VERBOSE)
    elif (FORMAT == 'HDF5'):
        d1 = pyTMD.spatial.from_HDF5(input_file,
            xname=VARIABLES[2], yname=VARIABLES[1],
            timename=VARIABLES[0], varname=VARIABLES[3],
            verbose=VERBOSE)
        d2 = pyTMD.spatial.from_HDF5(f2,
            xname=VARIABLES[2], yname=VARIABLES[1],
            timename=VARIABLES[0], varname='dem_h',
            verbose=VERBOSE)

    #-- grounded ice line string to determine if segment crosses coastline
    mpoly_obj,input_file,epsg = read_grounded_ice(base_dir, HEM, VARIABLES=[0])

    #-- converting x,y from projection to polar stereographic
    #-- could try to extract projection attributes from netCDF4 and HDF5 files
    try:
        crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(int(PROJECTION)))
    except (ValueError,pyproj.exceptions.CRSError):
        crs1 = pyproj.CRS.from_string(PROJECTION)
    crs2 = pyproj.CRS.from_string(epsg)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    X,Y = transformer.transform(d1['x'].flatten(),d1['y'].flatten())

    #-- densities of seawater and ice
    rho_w = 1030.0
    rho_ice = 917.0

    #-- find valid points with GZ for heights and interpolated DEM
    valid, = np.nonzero((~d1.mask) & (~d2.mask))
    #-- plot heights and DEM values
    plt.plot(d1['data'][valid],'r')
    plt.plot(d2['data'][valid],'b')
    #-- compress list (separate geosegs into sets of ranges)
    ice_gz_indices = compress_list(valid,10)
    for imin,imax in ice_gz_indices:
        #-- find valid indices within range
        i = sorted(set(np.arange(imin,imax+1)) & set(valid))
        #-- shapely LineString object for segment
        segment_line = shapely.geometry.LineString(np.c_[X[i],Y[i]])
        #-- determine if line segment intersects previously known GZ
        if segment_line.intersects(mpoly_obj[0]):
            #-- horizontal eulerian distance from start of segment
            dist = np.sqrt((X-X[0])**2 + (Y-Y[0])**2)
            #-- land ice height for grounding zone
            h_gz = d1['data'][i]
            #-- mean land ice height from digital elevation model
            h_mean = d2['data'][i]
            #-- deflection from mean land ice height in grounding zone
            dh_gz = h_gz - h_mean
            #-- quasi-freeboard: WGS84 elevation - geoid height
            QFB = h_gz# - geoid_h
            #-- ice thickness from quasi-freeboard and densities
            w_thick = QFB*rho_w/(rho_w-rho_ice)
            #-- fit with a hard piecewise model to get rough estimate of GZ
            C1,C2,PWMODEL = piecewise_fit(dist, dh_gz, STEP=5, CONF=0.95)
            #-- distance from estimated grounding line (0 = grounding line)
            d = (dist - C1[0]).astype(np.int)
            #-- determine if spacecraft is approaching coastline
            sco = True if np.sum(h_gz[d < 0]) < np.sum(h_gz[d > 0]) else False
            #-- fit physical elastic model
            PGZ,PA,PE,PT,PdH,PEMODEL = physical_elastic_model(dist, dh_gz,
                GZ=C1, ORIENTATION=sco, THICKNESS=w_thick, CONF=0.95)
            #-- linearly interpolate distance to grounding line
            XGZ = np.interp(PGZ[0],dist,X)
            YGZ = np.interp(PGZ[0],dist,X)
            GZI = np.interp(PGZ[0],dist,i)
            print(XGZ, YGZ, PGZ[0], PGZ[1], PT)
            plt.axvline(GZI)
    plt.show()

#-- Main program that calls calculate_grounding_zone()
def main():
    #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Calculates ice sheet grounding zones
            """
    )
    #-- command line options
    #-- input and output file
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='?',
        help='Input file')
    parser.add_argument('outfile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='?',
        help='Output file')
    #-- working data directory for shapefiles
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    #-- Digital elevation model (REMA, ArcticDEM, GIMP) to run
    #-- set the DEM model to run for a given granule (else set automatically)
    parser.add_argument('--model','-m',
        metavar='DEM', type=str, choices=('REMA', 'ArcticDEM', 'GIMP'),
        help='Digital Elevation Model to run')
    #-- input and output data format
    parser.add_argument('--format','-F',
        type=str, default='csv', choices=('csv','netCDF4','HDF5','geotiff'),
        help='Input and output data format')
    #-- variable names (for csv names of columns)
    parser.add_argument('--variables','-v',
        type=str, nargs='+', default=['time','lat','lon','data'],
        help='Variable names of data in input file')
    #-- number of header lines for csv files
    parser.add_argument('--header','-H',
        type=int, default=0,
        help='Number of header lines for csv files')
    #-- spatial projection (EPSG code or PROJ4 string)
    parser.add_argument('--projection','-P',
        type=str, default='4326',
        help='Spatial projection as EPSG code or PROJ4 string')
    #-- verbose output of processing run
    #-- print information about each input and output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    #-- permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of output file')
    args,_ = parser.parse_known_args()

    #-- set output file from input filename if not entered
    if not args.outfile:
        fileBasename,fileExtension = os.path.splitext(args.infile)
        vars = (fileBasename,'gz',fileExtension)
        args.outfile = '{0}_{1}{2}'.format(*vars)

    #-- run grounding zone program for input file
    calculate_grounding_zone(args.directory, args.infile, args.outfile,
        DEM_MODEL=args.model, FORMAT=args.format, VARIABLES=args.variables,
        HEADER=args.header, PROJECTION=args.projection, VERBOSE=args.verbose,
        MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
