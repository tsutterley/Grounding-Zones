#!/usr/bin/env python
u"""
compute_SET_ICESat_GLA12.py
Written by Tyler Sutterley (05/2023)
Calculates radial olid Earth tide displacements for correcting
    ICESat/GLAS L2 GLA12 Antarctic and Greenland Ice Sheet
    elevation data following IERS Convention (2010) guidelines

COMMAND LINE OPTIONS:
    -p X, --tide-system X: Permanent tide system for output values
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
    predict.py: calculates solid Earth tides

UPDATE HISTORY:
    Updated 05/2023: use timescale class for time conversion operations
        add option for using higher resolution ephemerides from JPL
        using pathlib to define and operate on paths
    Updated 04/2023: added permanent tide system offset (free-to-mean)
    Written 03/2023
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import warnings
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("h5py not available", ImportWarning)
try:
    import pyTMD
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyTMD not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12) from NSIDC
# compute solid Earth tide radial displacements at points and times
def compute_SET_ICESat(INPUT_FILE, TIDE_SYSTEM=None, EPHEMERIDES=None,
    VERBOSE=False, MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd',level=loglevel)

    # get directory from INPUT_FILE
    INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()
    logger.info(f'{str(INPUT_FILE)} -->')

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
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = \
            rx.findall(INPUT_FILE.name).pop()
    except (ValueError, IndexError):
        # output solid earth tide HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_SET{INPUT_FILE.suffix}'
    else:
        # output solid earth tide HDF5 file for NSIDC granules
        args = (PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_SET_{2}{3}{4}_{5}_{6}_{7}_{8}_{9}.h5'
        FILENAME = file_format.format(*args)
    # full path to output file
    OUTPUT_FILE = INPUT_FILE.with_name(FILENAME)

    # read GLAH12 HDF5 file
    fileID = h5py.File(INPUT_FILE, mode='r')
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

    # create timescale from J2000: seconds since 2000-01-01 12:00:00 UTC
    timescale = pyTMD.time.timescale().from_deltatime(DS_UTCTime_40HZ[:],
        epoch=pyTMD.time._j2000_epoch, standard='UTC')
    # convert tide times to dynamical time
    tide_time = timescale.tide + timescale.tt_ut1

    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = pyTMD.constants('TOPEX')
    wgs84 = pyTMD.constants('WGS84')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ, elev_40HZ = pyTMD.spatial.convert_ellipsoid(
        lat_TPX, elev_TPX,
        topex.a_axis, topex.flat,
        wgs84.a_axis, wgs84.flat,
        eps=1e-12, itmax=10)

    # convert input coordinates to cartesian
    X, Y, Z = pyTMD.spatial.to_cartesian(lon_40HZ, lat_40HZ, h=elev_40HZ,
        a_axis=wgs84.a_axis, flat=wgs84.flat)
    # compute ephemerides for lunisolar coordinates
    if (EPHEMERIDES.lower() == 'approximate'):
        # get low-resolution solar and lunar ephemerides
        SX, SY, SZ = pyTMD.astro.solar_ecef(timescale.MJD)
        LX, LY, LZ = pyTMD.astro.lunar_ecef(timescale.MJD)
    elif (EPHEMERIDES.upper() == 'JPL'):
        # compute solar and lunar ephemerides from JPL kernel
        SX, SY, SZ = pyTMD.astro.solar_ephemerides(timescale.MJD)
        LX, LY, LZ = pyTMD.astro.lunar_ephemerides(timescale.MJD)
    # convert coordinates to column arrays
    XYZ = np.c_[X, Y, Z]
    SXYZ = np.c_[SX, SY, SZ]
    LXYZ = np.c_[LX, LY, LZ]
    # predict solid earth tides (cartesian)
    dxi = pyTMD.predict.solid_earth_tide(tide_time,
        XYZ, SXYZ, LXYZ, a_axis=wgs84.a_axis,
        tide_system=TIDE_SYSTEM)
    # calculate radial component of solid earth tides
    dln, dlt, drad = pyTMD.spatial.to_geodetic(
        X + dxi[:,0], Y + dxi[:,1], Z + dxi[:,2],
        a_axis=wgs84.a_axis, flat=wgs84.flat)
    # remove effects of original topography
    tide_se = np.ma.zeros((n_40HZ),fill_value=fv)
    tide_se.data[:] = drad - elev_40HZ
    # replace fill values
    tide_se.mask = np.isnan(tide_se.data) | (elev_40HZ == fv)
    tide_se.data[tide_se.mask] = tide_se.fill_value
    # calculate permanent tide offset (meters)
    tide_se_free2mean = 0.06029 - \
        0.180873*np.sin(lat_40HZ*np.pi/180.0)**2

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
    IS_gla12_tide_attrs['lineage'] = INPUT_FILE.name
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
    # computed solid earth tide
    IS_gla12_tide['Data_40HZ']['Geophysical']['d_erElv'] = tide_se
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_erElv'] = tide_se.fill_value
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erElv'] = {}
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erElv']['units'] = "meters"
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erElv']['long_name'] = \
        "Solid Earth Tide Elevation"
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erElv']['description'] = \
        f'Solid earth tides in the {TIDE_SYSTEM} system'
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erElv']['reference'] = \
        'https://doi.org/10.1029/97JB01515'
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erElv']['coordinates'] = \
        "../DS_UTCTime_40"
    # computed solid earth permanent tide offset
    IS_gla12_tide['Data_40HZ']['Geophysical']['d_erf2mElv'] = tide_se_free2mean
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_erf2mElv'] = None
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erf2mElv'] = {}
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erf2mElv']['units'] = "meters"
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erf2mElv']['long_name'] = \
        "Solid Earth Tide Free-to-Mean conversion"
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erf2mElv']['description'] = \
        ('Additive value to convert solid earth tide from the tide_free system to '
         'the mean_tide system')
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erf2mElv']['reference'] = \
        'https://doi.org/10.1029/97JB01515'
    IS_gla12_tide_attrs['Data_40HZ']['Geophysical']['d_erf2mElv']['coordinates'] = \
        "../DS_UTCTime_40"

    # close the input HDF5 file
    fileID.close()

    # print file information
    logger.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_GLA12_tide_write(IS_gla12_tide, IS_gla12_tide_attrs,
        FILENAME=OUTPUT_FILE,
        FILL_VALUE=IS_gla12_fill, CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(MODE)

# PURPOSE: outputting the tide values for ICESat data to HDF5
def HDF5_GLA12_tide_write(IS_gla12_tide, IS_gla12_attrs,
    FILENAME='', FILL_VALUE=None, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    fileID = h5py.File(FILENAME, clobber)
    # create 40HZ HDF5 records
    h5 = dict(Data_40HZ={})

    # add HDF5 file attributes
    attrs = {a:v for a,v in IS_gla12_attrs.items() if not isinstance(v,dict)}
    for att_name,att_val in attrs.items():
       fileID.attrs[att_name] = att_val

    # add software information
    fileID.attrs['software_reference'] = pyTMD.version.project_name
    fileID.attrs['software_version'] = pyTMD.version.full_version

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
    h5['Data_40HZ']['DS_UTCTime_40'] = fileID.create_dataset(
        'Data_40HZ/DS_UTCTime_40', np.shape(val),
        data=val, dtype=val.dtype, compression='gzip')
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
        fileID.create_group(f'Data_40HZ/{group}')
        # add HDF5 group attributes
        for att_name,att_val in IS_gla12_attrs['Data_40HZ'][group].items():
            if not isinstance(att_val,dict):
                fileID['Data_40HZ'][group].attrs[att_name] = att_val
        # for each variable in the group
        for key,val in IS_gla12_tide['Data_40HZ'][group].items():
            fillvalue = FILL_VALUE['Data_40HZ'][group][key]
            attrs = IS_gla12_attrs['Data_40HZ'][group][key]
            # Defining the HDF5 dataset variables
            var = f'Data_40HZ/{group}/{key}'
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
        description="""Calculates radial solid Earth tide displacements for
            correcting ICESat/GLAS L2 GLA12 Antarctic and Greenland Ice Sheet
            elevation data following IERS Convention (2010) guidelines
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat GLA12 file to run')
    # permanent tide system for output values
    parser.add_argument('--tide-system','-p',
        type=str, choices=('tide_free','mean_tide'), default='tide_free',
        help='Permanent tide system for output values')
    # method for calculating lunisolar ephemerides
    parser.add_argument('--ephemerides','-c',
        type=str, choices=('approximate','JPL'), default='approximate',
        help='Method for calculating lunisolar ephemerides')
    # verbosity settings
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
        compute_SET_ICESat(FILE,
            TIDE_SYSTEM=args.tide_system,
            EPHEMERIDES=args.ephemerides,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
