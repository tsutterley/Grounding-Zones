#!/usr/bin/env python
u"""
interp_DAC_ICESat_GLAH12.py
Written by Tyler Sutterley (11/2022)
Calculates and interpolates dynamic atmospheric corrections for ICESat/GLAS
    L2 GLA12 Antarctic and Greenland Ice Sheet elevation data

Data will be interpolated for all valid points
    (masking land values will be needed for accurate assessments)

https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/
    atmospheric-corrections.html

Note that the AVISO DAC data are bz2 compressed netCDF4 files

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations
    spatial.py: utilities for reading and writing spatial data
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 11/2022: use f-strings for formatting verbose or ascii output
    Updated 05/2022: use argparse descriptions within sphinx documentation
    Updated 10/2021: using python logging for handling verbose output
        added parsing for converting file lines to arguments
    Updated 05/2021: print full path of output filename
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 09/2017: reduce grid domains for faster processing times
    Written 09/2017
"""
from __future__ import print_function

import os
import re
import bz2
import pyproj
import logging
import netCDF4
import argparse
import warnings
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("h5py not available")
try:
    import icesat2_toolkit as is2tk
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("icesat2_toolkit not available")
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12) from NSIDC
# calculate and interpolate the dynamic atmospheric correction
def interp_DAC_ICESat_GLAH12(base_dir, INPUT_FILE, VERBOSE=False, MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # get directory from INPUT_FILE
    logging.info(f'{INPUT_FILE} -->')
    DIRECTORY = os.path.dirname(INPUT_FILE)

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
        # output dynamic atmospheric correction HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        args = (fileBasename,fileExtension)
        OUTPUT_FILE = '{0}_DAC{1}'.format(*args)
    else:
        # output dynamic atmospheric correction HDF5 file for NSIDC granules
        args = (PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_DAC_{2}{3}{4}_{5}_{6}_{7}_{8}_{9}.h5'
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

    # convert time from J2000 to days relative to 1950-01-01 (MJD:33282)
    # J2000: seconds since 2000-01-01 12:00:00 UTC
    t = DS_UTCTime_40HZ/86400.0 + 18262.5
    # days and hours to read
    unique_hours = np.unique([np.floor(t*24.0/6.0)*6.0, np.ceil(t*24.0/6.0)*6.0])
    days,hours = (unique_hours // 24, unique_hours % 24)

    # semimajor axis (a) and flattening (f) for TP and WGS84 ellipsoids
    atop,ftop = (6378136.3,1.0/298.257)
    awgs,fwgs = (6378137.0,1.0/298.257223563)
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ,elev_40HZ = is2tk.spatial.convert_ellipsoid(lat_TPX,
        elev_TPX, atop, ftop, awgs, fwgs, eps=1e-12, itmax=10)

    # pyproj transformer for converting from input coordinates (EPSG)
    # to model coordinates
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_string('+proj=longlat +ellps=WGS84 +datum=WGS84 '
        '+no_defs lon_wrap=180')
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(lon_40HZ, lat_40HZ)

    # shape of pressure field
    ny,nx = (721,1440)
    # allocate for DAC fields
    idac = np.ma.zeros((len(days),ny,nx))
    icjd = np.zeros((len(days)))
    for i,CJD in enumerate(days):
        # convert from CNES Julians Day to calendar
        YY,MM,DD,HH,MN,SS = is2tk.time.convert_julian(CJD + 2433282.5,
            format='tuple')
        # input file for 6-hour period
        input_file = f'dac_dif_{CJD:0.0f}_{hours[i]:02.0f}.nc.bz2'
        # read bytes from compressed file
        fd = bz2.BZ2File(os.path.join(base_dir,f'{YY:0.0f}',input_file))
        # read netCDF file for time
        with netCDF4.Dataset('dac', mode='r', memory=fd.read()) as fid:
            ilon = fid['longitude'][:]
            ilat = fid['latitude'][:]
            idac[i,:,:] = fid['dac'][:]
            icjd[i] = fid['dac'].getncattr('Date_CNES_JD')
        # close the compressed file objects
        fd.close()
    # create an interpolator for dynamic atmospheric correction
    RGI = scipy.interpolate.RegularGridInterpolator((icjd,ilat,ilon), idac,
        bounds_error=False)
    # interpolate dynamic atmospheric correction to points
    DAC = np.ma.zeros((rec_ndx_40HZ), fill_value=fv)
    DAC.data = RGI.__call__(np.c_[t, iy, ix])
    DAC.mask = np.isnan(DAC.data)
    DAC.data[DAC.mask] = DAC.fill_value

    # copy variables for outputting to HDF5 file
    IS_gla12_corr = dict(Data_40HZ={})
    IS_gla12_fill = dict(Data_40HZ={})
    IS_gla12_corr_attrs = dict(Data_40HZ={})

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
        IS_gla12_corr_attrs[att] = fileID.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_corr_attrs['Campaign'] = fileID['ANCILLARY_DATA'].attrs['Campaign']

    # add attributes for input GLA12 file
    IS_gla12_corr_attrs['input_files'] = os.path.basename(INPUT_FILE)
    # update geospatial ranges for ellipsoid
    IS_gla12_corr_attrs['geospatial_lat_min'] = np.min(lat_40HZ)
    IS_gla12_corr_attrs['geospatial_lat_max'] = np.max(lat_40HZ)
    IS_gla12_corr_attrs['geospatial_lon_min'] = np.min(lon_40HZ)
    IS_gla12_corr_attrs['geospatial_lon_max'] = np.max(lon_40HZ)
    IS_gla12_corr_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_corr_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_corr_attrs['geospatial_ellipsoid'] = "WGS84"

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_corr_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time, geolocation and geophysical groups
    for var in ['Time','Geolocation','Geophysical']:
        IS_gla12_corr['Data_40HZ'][var] = {}
        IS_gla12_fill['Data_40HZ'][var] = {}
        IS_gla12_corr_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_corr_attrs['Data_40HZ'][var][att_name] = att_val

    # J2000 time
    IS_gla12_corr['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_fill['Data_40HZ']['DS_UTCTime_40'] = None
    IS_gla12_corr_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in fileID['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_corr_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    # record
    IS_gla12_corr['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_fill['Data_40HZ']['Time']['i_rec_ndx'] = None
    IS_gla12_corr_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_corr_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    # latitude
    IS_gla12_corr['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lat'] = None
    IS_gla12_corr_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_corr_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    # longitude
    IS_gla12_corr['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lon'] = None
    IS_gla12_corr_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_corr_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    # dynamic atmospheric correction (DAC)
    IS_gla12_corr['Data_40HZ']['Geophysical']['d_dacElv'] = DAC.copy()
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_dacElv'] = DAC.fill_value
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_dacElv'] = {}
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_dacElv']['units'] = "meters"
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_dacElv']['long_name'] = \
        "Dynamic_Atmosphere_Correction"
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_dacElv']['description'] = ("Dynamic_"
        "atmospheric_correction_(DAC)_which_includes_inverse_barometer_(IB)_effects")
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_dacElv']['source'] = \
        'Mog2D-G_High_Resolution_barotropic_model'
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_dacElv']['reference'] = \
        ('https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/'
         'atmospheric-corrections.html')
    IS_gla12_corr_attrs['Data_40HZ']['Geophysical']['d_dacElv']['coordinates'] = \
        "../DS_UTCTime_40"

    # close the input HDF5 file
    fileID.close()

    # print file information
    logging.info(f'\t{OUTPUT_FILE}')
    HDF5_GLA12_corr_write(IS_gla12_corr, IS_gla12_corr_attrs,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE),
        FILL_VALUE=IS_gla12_fill, CLOBBER=True)
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

# PURPOSE: outputting the correction values for ICESat data to HDF5
def HDF5_GLA12_corr_write(IS_gla12_tide, IS_gla12_attrs,
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
    fileID.attrs['software_revision'] = gz.utilities.get_git_revision_hash()

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
        description="""Calculates and interpolates dynamic atmospheric
            corrections to ICESat/GLAS L2 GLA12 Antarctic and Greenland
            Ice Sheet elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat GLA12 file to run')
    # directory with reanalysis data
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
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

    # run for each input GLAH12 file
    for FILE in args.infile:
        interp_DAC_ICESat_GLAH12(args.directory, FILE,
            VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
