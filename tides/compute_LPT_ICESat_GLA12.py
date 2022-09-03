#!/usr/bin/env python
u"""
compute_LPT_ICESat_GLA12.py
Written by Tyler Sutterley (07/2022)
Calculates radial load pole tide displacements for correcting ICESat/GLAS
    L2 GLA12 Antarctic and Greenland Ice Sheet elevation data following
    IERS Convention (2010) guidelines

COMMAND LINE OPTIONS:
    -M X, --mode X: Permission mode of directories and files created
    -V, --verbose: Output information about each created file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations
    spatial.py: utilities for reading, writing and operating on spatial data
    utilities.py: download and management utilities for syncing files
    iers_mean_pole.py: provides the angular coordinates of IERS Mean Pole
    read_iers_EOP.py: read daily earth orientation parameters from IERS

REFERENCES:
    S Desai, "Observing the pole tide with satellite altimetry", Journal of
        Geophysical Research: Oceans, 107(C11), 2002. doi: 10.1029/2001JC001224
    S Desai, J Wahr and B Beckley "Revisiting the pole tide for and from
        satellite altimetry", Journal of Geodesy, 89(12), p1233-1243, 2015.
        doi: 10.1007/s00190-015-0848-7

UPDATE HISTORY:
    Updated 07/2022: place some imports within try/except statements
    Updated 04/2022: use argparse descriptions within documentation
    Updated 02/2022: save ICESat campaign attribute to output file
    Updated 10/2021: using python logging for handling verbose output
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 04/2021: can use a generically named GLA12 file as input
    Updated 03/2021: use cartesian coordinate conversion routine in spatial
    Updated 12/2020: H5py deprecation warning change to use make_scale
        merged time conversion routines into module
    Written 12/2020
"""
from __future__ import print_function

import sys
import os
import re
import logging
import argparse
import warnings
import numpy as np
import scipy.interpolate
import pyTMD.time
import pyTMD.spatial
import pyTMD.utilities
from pyTMD.iers_mean_pole import iers_mean_pole
from pyTMD.read_iers_EOP import read_iers_EOP
# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("h5py not available")
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12) from NSIDC
# compute load pole tide radial displacements at points and times
def compute_LPT_ICESat(FILE, VERBOSE=False, MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd',level=loglevel)

    # get directory from FILE
    logger.info('{0} -->'.format(FILE))
    DIRECTORY = os.path.dirname(FILE)

    # compile regular expression operator for extracting information from file
    rx = re.compile((r'GLAH(\d{2})_(\d{3})_(\d{1})(\d{1})(\d{2})_(\d{3})_'
        r'(\d{4})_(\d{1})_(\d{2})_(\d{4})\.H5'), re.VERBOSE)
    # extract parameters from ICESat/GLAS HDF5 file name
    # PRD:  Product number (01, 05, 06, 12, 13, 14, or 15)
    # RL:  Release number for process that created the product = 634
    # RGTP:  Repeat ground-track phase (1=8-day, 2=91-day, 3=transfer orbit)
    # ORB:   Reference orbit number (starts at 1 and increments each time a
    #           new reference orbit ground track file is obtained.)
    # INST:  Instance number (increments every time the satellite enters a
    #           different reference orbit)
    # CYCL:   Cycle of reference orbit for this phase
    # TRK: Track within reference orbit
    # SEG:   Segment of orbit
    # GRAN:  Granule version number
    # TYPE:  File type
    try:
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = rx.findall(FILE).pop()
    except:
        # output load pole tide HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(FILE)
        OUTPUT_FILE = '{0}_{1}{2}'.format(fileBasename,'LPT',fileExtension)
    else:
        # output load pole tide HDF5 file for NSIDC granules
        args = (PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_LPT_{2}{3}{4}_{5}_{6}_{7}_{8}_{9}.h5'
        OUTPUT_FILE = file_format.format(*args)

    # read GLAH12 HDF5 file
    fileID = h5py.File(FILE,'r')
    n_40HZ, = fileID['Data_40HZ']['Time']['i_rec_ndx'].shape
    # get variables and attributes
    rec_ndx_40HZ = fileID['Data_40HZ']['Time']['i_rec_ndx'][:].copy()
    # seconds since 2000-01-01 12:00:00 UTC (J2000)
    DS_UTCTime_40HZ = fileID['Data_40HZ']['DS_UTCTime_40'][:].copy()
    # Latitude (degrees North)
    lat_TPX = fileID['Data_40HZ']['Geolocation']['d_lat'][:].copy()
    # Longitude (degrees East)
    lon_40HZ = fileID['Data_40HZ']['Geolocation']['d_lon'][:].copy()
    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
    elev_TPX = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'][:].copy()
    fv = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'].attrs['_FillValue']

    # convert time from UTC time of day to Modified Julian Days (MJD)
    # J2000: seconds since 2000-01-01 12:00:00 UTC
    t = DS_UTCTime_40HZ[:]/86400.0 + 51544.5
    # convert from MJD to calendar dates
    YY,MM,DD,HH,MN,SS = pyTMD.time.convert_julian(t + 2400000.5,format='tuple')
    # convert calendar dates into year decimal
    tdec = pyTMD.time.convert_calendar_decimal(YY,MM,day=DD,
        hour=HH,minute=MN,second=SS)

    # semimajor axis (a) and flattening (f) for TP and WGS84 ellipsoids
    atop,ftop = (6378136.3,1.0/298.257)
    awgs,fwgs = (6378137.0,1.0/298.257223563)
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ,elev_40HZ = pyTMD.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        atop, ftop, awgs, fwgs, eps=1e-12, itmax=10)

    # degrees to radians
    dtr = np.pi/180.0
    atr = np.pi/648000.0
    # earth and physical parameters (IERS and WGS84)
    G = 6.67428e-11# universal constant of gravitation [m^3/(kg*s^2)]
    GM = 3.986004418e14# geocentric gravitational constant [m^3/s^2]
    ge = 9.7803278# mean equatorial gravity [m/s^2]
    a_axis = 6378136.6# semimajor axis of the WGS84 ellipsoid [m]
    flat = 1.0/298.257223563# flattening of the WGS84 ellipsoid
    b_axis = (1.0 -flat)*a_axis# semiminor axis of the WGS84 ellipsoid [m]
    omega = 7.292115e-5# mean rotation rate of the Earth [radians/s]
    # tidal love number appropriate for the load tide
    hb2 = 0.6207
    # Linear eccentricity, first and second numerical eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    ecc1 = lin_ecc/a_axis
    ecc2 = lin_ecc/b_axis
    # m parameter [omega^2*a^2*b/(GM)]. p. 70, Eqn.(2-137)
    m = omega**2*((1 -flat)*a_axis**3)/GM
    # flattening components
    f_2 = -flat + (5.0/2.0)*m + (1.0/2.0)*flat**2.0 - (26.0/7.0)*flat*m + \
        (15.0/4.0)*m**2.0
    f_4 = -(1.0/2.0)*flat**2.0 + (5.0/2.0)*flat*m

    # convert from geodetic latitude to geocentric latitude
    # calculate X, Y and Z from geodetic latitude and longitude
    X,Y,Z = pyTMD.spatial.to_cartesian(lon_40HZ,lat_40HZ,h=elev_40HZ,
        a_axis=a_axis,flat=flat)
    rr = np.sqrt(X**2.0 + Y**2.0 + Z**2.0)
    # calculate geocentric latitude and convert to degrees
    latitude_geocentric = np.arctan(Z / np.sqrt(X**2.0 + Y**2.0))/dtr
    # colatitude and longitude in radians
    theta = dtr*(90.0 - latitude_geocentric)
    phi = lon_40HZ*dtr

    # compute normal gravity at spatial location and elevation of points.
    # normal gravity at the equator. p. 79, Eqn.(2-186)
    gamma_a = (GM/(a_axis*b_axis)) * (1.0-(3.0/2.0)*m - (3.0/14.0)*ecc2**2.0*m)
    # Normal gravity. p. 80, Eqn.(2-199)
    gamma_0 = gamma_a*(1.0 + f_2*np.cos(theta)**2.0 +
        f_4*np.sin(np.pi*latitude_geocentric/180.0)**4.0)
    # Normal gravity at height h. p. 82, Eqn.(2-215)
    gamma_h = gamma_0*(1.0 -
        (2.0/a_axis)*(1.0+flat+m-2.0*flat*np.cos(theta)**2.0)*elev_40HZ + \
        (3.0/a_axis**2.0)*elev_40HZ**2.0)

    # pole tide files (mean and daily)
    mean_pole_file = pyTMD.utilities.get_data_path(['data','mean-pole.tab'])
    pole_tide_file = pyTMD.utilities.get_data_path(['data','finals.all'])
    # read IERS daily polar motion values
    EOP = read_iers_EOP(pole_tide_file)
    # create cubic spline interpolations of daily polar motion values
    xSPL = scipy.interpolate.UnivariateSpline(EOP['MJD'],EOP['x'],k=3,s=0)
    ySPL = scipy.interpolate.UnivariateSpline(EOP['MJD'],EOP['y'],k=3,s=0)

    # calculate angular coordinates of mean pole at time tdec
    mpx,mpy,fl = iers_mean_pole(mean_pole_file,tdec,'2015')
    # interpolate daily polar motion values to time using cubic splines
    px = xSPL(t)
    py = ySPL(t)
    # calculate differentials from mean pole positions
    mx = px - mpx
    my = -(py - mpy)
    # calculate radial displacement at time
    dfactor = -hb2*atr*(omega**2*rr**2)/(2.0*gamma_h)
    Srad = np.ma.zeros((n_40HZ),fill_value=fv)
    Srad.data[:] = dfactor*np.sin(2.0*theta)*(mx*np.cos(phi) + my*np.sin(phi))
    # replace fill values
    Srad.mask = np.isnan(Srad.data)
    Srad.data[Srad.mask] = Srad.fill_value

    # copy variables for outputting to HDF5 file
    IS_gla12_tide = dict(Data_40HZ={})
    IS_gla12_fill = dict(Data_40HZ={})
    IS_gla12_tide_attrs = dict(Data_40HZ={})

    # copy global file attributes
    global_attribute_list = ['featureType','title','comment','summary','license',
        'references','AccessConstraints','CitationforExternalPublication',
        'contributor_role','contributor_name','creator_name','creator_email',
        'publisher_name','publisher_email','publisher_url','platform','instrument',
        'processing_level','date_created','spatial_coverage_type','history',
        'keywords','keywords_vocabulary','naming_authority','project','time_type',
        'date_type','time_coverage_start','time_coverage_end',
        'time_coverage_duration','source','HDFVersion','identifier_product_type',
        'identifier_product_format_version','Conventions','institution',
        'ReprocessingPlanned','ReprocessingActual','LocalGranuleID',
        'ProductionDateTime','LocalVersionID','PGEVersion','OrbitNumber',
        'StartOrbitNumber','StopOrbitNumber','EquatorCrossingLongitude',
        'EquatorCrossingTime','EquatorCrossingDate','ShortName','VersionID',
        'InputPointer','RangeBeginningTime','RangeEndingTime','RangeBeginningDate',
        'RangeEndingDate','PercentGroundHit','OrbitQuality','Cycle','Track',
        'Instrument_State','Timing_Bias','ReferenceOrbit','SP_ICE_PATH_NO',
        'SP_ICE_GLAS_StartBlock','SP_ICE_GLAS_EndBlock','Instance','Range_Bias',
        'Instrument_State_Date','Instrument_State_Time','Range_Bias_Date',
        'Range_Bias_Time','Timing_Bias_Date','Timing_Bias_Time',
        'identifier_product_doi','identifier_file_uuid',
        'identifier_product_doi_authority']
    for att in global_attribute_list:
        IS_gla12_tide_attrs[att] = fileID.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_tide_attrs['Campaign'] = fileID['ANCILLARY_DATA'].attrs['Campaign']

    # add attributes for input GLA12 file
    IS_gla12_tide_attrs['input_files'] = os.path.basename(FILE)
    # update geospatial ranges for ellipsoid
    IS_gla12_tide_attrs['geospatial_lat_min'] = np.min(lat_40HZ)
    IS_gla12_tide_attrs['geospatial_lat_max'] = np.max(lat_40HZ)
    IS_gla12_tide_attrs['geospatial_lon_min'] = np.min(lon_40HZ)
    IS_gla12_tide_attrs['geospatial_lon_max'] = np.max(lon_40HZ)
    IS_gla12_tide_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_tide_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_tide_attrs['geospatial_ellipsoid'] = "WGS84"

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_tide_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time, geolocation and geophysical groups
    for var in ['Time','Geolocation','Geophysical']:
        IS_gla12_tide['Data_40HZ'][var] = {}
        IS_gla12_fill['Data_40HZ'][var] = {}
        IS_gla12_tide_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_tide_attrs['Data_40HZ'][var][att_name] = att_val

    # J2000 time
    IS_gla12_tide['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_fill['Data_40HZ']['DS_UTCTime_40'] = None
    IS_gla12_tide_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in fileID['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_tide_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    # record
    IS_gla12_tide['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_fill['Data_40HZ']['Time']['i_rec_ndx'] = None
    IS_gla12_tide_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_tide_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    # latitude
    IS_gla12_tide['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lat'] = None
    IS_gla12_tide_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_tide_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    # longitude
    IS_gla12_tide['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lon'] = None
    IS_gla12_tide_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_tide_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    # geophysical variables
    # computed Solid Earth load pole tide
    IS_gla12_tide['Data_40HZ']['Geophysical']['d_poElv'] = Srad
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_poElv'] = Srad.fill_value
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_poElv'] = {}
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_poElv']['units'] = "meters"
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_poElv']['long_name'] = \
        "Solid Earth Pole Tide"
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_poElv']['description'] = ("Solid "
        "Earth pole tide radial displacements due to polar motion")
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_poElv']['reference'] = \
        'ftp://tai.bipm.org/iers/conv2010/chapter7/tn36_c7.pdf'
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_poElv']['coordinates'] = \
        "../DS_UTCTime_40"

    # close the input HDF5 file
    fileID.close()

    # print file information
    logger.info('\t{0}'.format(OUTPUT_FILE))
    HDF5_GLA12_tide_write(IS_gla12_tide, IS_gla12_tide_attrs,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE),
        FILL_VALUE=IS_gla12_fill, CLOBBER=True)
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

# PURPOSE: outputting the tide values for ICESat data to HDF5
def HDF5_GLA12_tide_write(IS_gla12_tide, IS_gla12_attrs,
    FILENAME='', FILL_VALUE=None, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)
    # create 40HZ HDF5 records
    h5 = dict(Data_40HZ={})

    # add HDF5 file attributes
    attrs = {a:v for a,v in IS_gla12_attrs.items() if not isinstance(v,dict)}
    for att_name,att_val in attrs.items():
       fileID.attrs[att_name] = att_val

    # create Data_40HZ group
    fileID.create_group('Data_40HZ')
    # add HDF5 40HZ group attributes
    for att_name,att_val in IS_gla12_attrs['Data_40HZ'].items():
        if att_name not in ('DS_UTCTime_40',) and not isinstance(att_val,dict):
            fileID['Data_40HZ'].attrs[att_name] = att_val

    # add 40HZ time variable
    val = IS_gla12_tide['Data_40HZ']['DS_UTCTime_40']
    attrs = IS_gla12_attrs['Data_40HZ']['DS_UTCTime_40']
    # Defining the HDF5 dataset variables
    var = '{0}/{1}'.format('Data_40HZ','DS_UTCTime_40')
    h5['Data_40HZ']['DS_UTCTime_40'] = fileID.create_dataset(var,
        np.shape(val), data=val, dtype=val.dtype, compression='gzip')
    # make dimension
    h5['Data_40HZ']['DS_UTCTime_40'].make_scale('DS_UTCTime_40')
    # add HDF5 variable attributes
    for att_name,att_val in attrs.items():
        h5['Data_40HZ']['DS_UTCTime_40'].attrs[att_name] = att_val

    # for each variable group
    for group in ['Time','Geolocation','Geophysical']:
        # add group to dict
        h5['Data_40HZ'][group] = {}
        # create Data_40HZ group
        fileID.create_group('Data_40HZ/{0}'.format(group))
        # add HDF5 group attributes
        for att_name,att_val in IS_gla12_attrs['Data_40HZ'][group].items():
            if not isinstance(att_val,dict):
                fileID['Data_40HZ'][group].attrs[att_name] = att_val
        # for each variable in the group
        for key,val in IS_gla12_tide['Data_40HZ'][group].items():
            fillvalue = FILL_VALUE['Data_40HZ'][group][key]
            attrs = IS_gla12_attrs['Data_40HZ'][group][key]
            # Defining the HDF5 dataset variables
            var = '{0}/{1}/{2}'.format('Data_40HZ',group,key)
            # use variable compression if containing fill values
            if fillvalue:
                h5['Data_40HZ'][group][key] = fileID.create_dataset(var,
                    np.shape(val), data=val, dtype=val.dtype,
                    fillvalue=fillvalue, compression='gzip')
            else:
                h5['Data_40HZ'][group][key] = fileID.create_dataset(var,
                    np.shape(val), data=val, dtype=val.dtype,
                    compression='gzip')
            # attach dimensions
            for i,dim in enumerate(['DS_UTCTime_40']):
                h5['Data_40HZ'][group][key].dims[i].attach_scale(
                    h5['Data_40HZ'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5['Data_40HZ'][group][key].attrs[att_name] = att_val

    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates radial load pole tide displacements for
            correcting ICESat/GLAS L2 GLA12 Antarctic and Greenland Ice Sheet
            elevation data following IERS Convention (2010) guidelines
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat GLA12 file to run')
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

    # run for each input GLA12 file
    for FILE in args.infile:
        compute_LPT_ICESat(FILE, VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
