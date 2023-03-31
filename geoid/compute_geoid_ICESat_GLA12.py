#!/usr/bin/env python
u"""
compute_geoid_ICESat_GLA12.py
Written by Tyler Sutterley (12/2022)
Computes geoid undulations for correcting ICESat/GLAS L2 GLA12
    Antarctic and Greenland Ice Sheet elevation data

INPUTS:
    input_file: ICESat GLA12 data file

COMMAND LINE OPTIONS:
    -G X, --gravity X: Gravity model file to use (.gfc format)
    -l X, --lmax X: maximum spherical harmonic degree (level of truncation)
    -n X, --love X: Degree 2 load Love number (default EGM2008 value)
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
    utilities.py: download and management utilities for syncing files
    geoid_undulation.py: geoidal undulation at a given latitude and longitude
    read_ICGEM_harmonics.py: reads the coefficients for a given gravity model file
    real_potential.py: real potential at a latitude and height for gravity model
    norm_potential.py: normal potential of an ellipsoid at a latitude and height
    norm_gravity.py: normal gravity of an ellipsoid at a latitude and height
    ref_ellipsoid.py: Computes parameters for a reference ellipsoid
    gauss_weights.py: Computes Gaussian weights as a function of degree

UPDATE HISTORY:
    Updated 12/2022: single implicit import of grounding zone tools
        use reference ellipsoid function from geoid toolkit for parameters
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
        additionally output conversion between tide free and mean tide values
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 04/2021: can use a generically named GLA12 file as input
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: H5py deprecation warning change to use make_scale
    Updated 10/2020: using argparse to set command line parameters
    Updated 08/2020: using python3 compatible regular expressions
    Updated 10/2019: changing Y/N flags to True/False
    Written 07/2017
"""
from __future__ import print_function

import sys
import os
import re
import logging
import argparse
import warnings
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import geoid_toolkit as geoidtk
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("geoid_toolkit not available", ImportWarning)
try:
    import h5py
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("h5py not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12) from NSIDC
# and computes geoid undulation at points
def compute_geoid_ICESat(model_file, INPUT_FILE, LMAX=None, LOVE=None,
    VERBOSE=False, MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # get directory from INPUT_FILE
    logging.info(f'{INPUT_FILE} -->')
    DIRECTORY = os.path.dirname(INPUT_FILE)

    # read gravity model Ylms and change tide to tide free
    Ylms = geoidtk.read_ICGEM_harmonics(model_file, LMAX=LMAX, TIDE='tide_free')
    R = np.float64(Ylms['radius'])
    GM = np.float64(Ylms['earth_gravity_constant'])
    LMAX = np.int64(Ylms['max_degree'])
    # reference to WGS84 ellipsoid
    REFERENCE = 'WGS84'

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
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = rx.findall(INPUT_FILE).pop()
    except:
        # output geoid HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        args = (fileBasename,Ylms['modelname'],fileExtension)
        OUTPUT_FILE = '{0}_{1}_GEOID{2}'.format(*args)
    else:
        # output geoid HDF5 file for NSIDC granules
        args = (PRD,RL,Ylms['modelname'],RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_{2}_GEOID_{3}{4}{5}_{6}_{7}_{8}_{9}_{10}.h5'
        OUTPUT_FILE = file_format.format(*args)

    # read GLAH12 HDF5 file
    fileID = h5py.File(INPUT_FILE,'r')
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

    # semimajor axis (a) and flattening (f) for TP and WGS84 ellipsoids
    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = geoidtk.ref_ellipsoid('TOPEX')
    wgs84 = geoidtk.ref_ellipsoid('WGS84')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ,elev_40HZ = geoidtk.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        topex['a'], topex['f'], wgs84['a'], wgs84['f'], eps=1e-12, itmax=10)
    # colatitude in radians
    theta_40HZ = (90.0 - lat_40HZ)*np.pi/180.0

    # calculate geoid at coordinates
    N = geoidtk.geoid_undulation(lat_40HZ, lon_40HZ, REFERENCE,
        Ylms['clm'], Ylms['slm'], LMAX, R, GM, GAUSS=0)
    # calculate offset for converting from tide_free to mean_tide
    # legendre polynomial of degree 2 (unnormalized)
    P2 = 0.5*(3.0*np.cos(theta_40HZ)**2 - 1.0)
    # from Rapp 1991 (Consideration of Permanent Tidal Deformation)
    free2mean = -0.198*P2*(1.0 + LOVE)

    # copy variables for outputting to HDF5 file
    IS_gla12_geoid = dict(Data_40HZ={})
    IS_gla12_fill = dict(Data_40HZ={})
    IS_gla12_geoid_attrs = dict(Data_40HZ={})

    # copy global file attributes of interest
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
        IS_gla12_geoid_attrs[att] = fileID.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_geoid_attrs['Campaign'] = fileID['ANCILLARY_DATA'].attrs['Campaign']

    # add attributes for input GLA12 file
    IS_gla12_geoid_attrs['input_files'] = os.path.basename(INPUT_FILE)
    # update geospatial ranges for ellipsoid
    IS_gla12_geoid_attrs['geospatial_lat_min'] = np.min(lat_40HZ)
    IS_gla12_geoid_attrs['geospatial_lat_max'] = np.max(lat_40HZ)
    IS_gla12_geoid_attrs['geospatial_lon_min'] = np.min(lon_40HZ)
    IS_gla12_geoid_attrs['geospatial_lon_max'] = np.max(lon_40HZ)
    IS_gla12_geoid_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_geoid_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_geoid_attrs['geospatial_ellipsoid'] = "WGS84"

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_geoid_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time, geolocation and geophysical groups
    for var in ['Time','Geolocation','Geophysical']:
        IS_gla12_geoid['Data_40HZ'][var] = {}
        IS_gla12_fill['Data_40HZ'][var] = {}
        IS_gla12_geoid_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_geoid_attrs['Data_40HZ'][var][att_name] = att_val

    # J2000 time
    IS_gla12_geoid['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_fill['Data_40HZ']['DS_UTCTime_40'] = None
    IS_gla12_geoid_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in fileID['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_geoid_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    # record
    IS_gla12_geoid['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_fill['Data_40HZ']['Time']['i_rec_ndx'] = None
    IS_gla12_geoid_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    IS_gla12_geoid_attrs['Data_40HZ']['Time']['i_rec_ndx']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_geoid_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    # latitude
    IS_gla12_geoid['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lat'] = None
    IS_gla12_geoid_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    IS_gla12_geoid_attrs['Data_40HZ']['Geolocation']['d_lat']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_geoid_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    # longitude
    IS_gla12_geoid['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lon'] = None
    IS_gla12_geoid_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    IS_gla12_geoid_attrs['Data_40HZ']['Geolocation']['d_lon']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_geoid_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    # geophysical variables
    # geoid undulation
    IS_gla12_geoid['Data_40HZ']['Geophysical']['d_gdHt'] = N.astype(np.float64)
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_gdHt'] = None
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt'] = {}
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['units'] = "meters"
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['long_name'] = 'Geoidal_Undulation'
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['description'] = ('Geoidal '
        f'undulation above the {REFERENCE} ellipsoid')
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['tide_system'] = Ylms['tide_system']
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['source'] = Ylms['modelname']
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['earth_gravity_constant'] = GM
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['radius'] = R
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['degree_of_truncation'] = LMAX
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdHt']['coordinates'] = \
        "../DS_UTCTime_40"
    # geoid conversion
    IS_gla12_geoid['Data_40HZ']['Geophysical']['d_gdfree2mean'] = free2mean.copy()
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_gdfree2mean'] = None
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdfree2mean'] = {}
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdfree2mean']['units'] = "meters"
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdfree2mean']['long_name'] = ('Geoid_'
        'Free-to-Mean_conversion')
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdfree2mean']['description'] = ('Additive '
        'value to convert geoid heights from the tide-free system to the mean-tide system')
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdfree2mean']['earth_gravity_constant'] = GM
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdfree2mean']['radius'] = R
    IS_gla12_geoid_attrs['Data_40HZ']['Geophysical']['d_gdfree2mean']['coordinates'] = \
        "../DS_UTCTime_40"

    # close the input HDF5 file
    fileID.close()

    # print file information
    logging.info(f'\t{os.path.join(DIRECTORY,OUTPUT_FILE)}')
    HDF5_GLA12_geoid_write(IS_gla12_geoid, IS_gla12_geoid_attrs,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE),
        FILL_VALUE=IS_gla12_fill, CLOBBER=True)
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

# PURPOSE: outputting the geoid values for ICESat data to HDF5
def HDF5_GLA12_geoid_write(IS_gla12_geoid, IS_gla12_attrs,
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

    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version

    # create Data_40HZ group
    fileID.create_group('Data_40HZ')
    # add HDF5 40HZ group attributes
    for att_name,att_val in IS_gla12_attrs['Data_40HZ'].items():
        if att_name not in ('DS_UTCTime_40',) and not isinstance(att_val,dict):
            fileID['Data_40HZ'].attrs[att_name] = att_val

    # add 40HZ time variable
    val = IS_gla12_geoid['Data_40HZ']['DS_UTCTime_40']
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
        for key,val in IS_gla12_geoid['Data_40HZ'][group].items():
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
        description="""Calculates geoid undunations for correcting ICESat/GLAS
            L2 GLA12 Antarctic and Greenland Ice Sheet elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    # input ICESat GLAS files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat GLA12 file to run')
    # set gravity model file to use
    parser.add_argument('--gravity','-G',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Gravity model file to use')
    # maximum spherical harmonic degree (level of truncation)
    parser.add_argument('--lmax','-l',
        type=int, help='Maximum spherical harmonic degree')
    # load love number of degree 2 (default EGM2008 value)
    parser.add_argument('--love','-n',
        type=float, default=0.3,
        help='Degree 2 load Love number')
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
        compute_geoid_ICESat(args.gravity, FILE, LMAX=args.lmax,
            LOVE=args.love, VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
