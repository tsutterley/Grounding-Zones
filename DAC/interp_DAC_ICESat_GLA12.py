#!/usr/bin/env python
u"""
interp_DAC_ICESat_GLA12.py
Written by Tyler Sutterley (06/2024)
Interpolates AVISO dynamic atmospheric corrections (DAC) for ICESat/GLAS
    L2 GLA12 Antarctic and Greenland Ice Sheet elevation data

Data will be interpolated for all valid points
    (masking land values will be needed for accurate assessments)

https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/
    atmospheric-corrections.html

Note that the AVISO DAC data can be bz2 compressed netCDF4 files

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
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
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    spatial.py: utilities for reading and writing spatial data
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 06/2024: fix case where input files are not bz2 compressed
    Updated 05/2024: use wrapper to importlib for optional dependencies
        fix memory allocation for output 40HZ data
        use ellipsoid transformation function from pyTMD
    Updated 04/2024: use timescale for temporal operations
    Updated 08/2023: create s3 filesystem when using s3 urls as input
    Updated 12/2022: single implicit import of grounding zone tools
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

import re
import bz2
import logging
import pathlib
import argparse
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
netCDF4 = gz.utilities.import_dependency('netCDF4')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12)
# calculate and interpolate the dynamic atmospheric correction
def interp_DAC_ICESat_GLA12(base_dir, INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # directory setup
    base_dir = pathlib.Path(base_dir).expanduser().absolute()

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
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = rx.findall(GRANULE).pop()
    except:
        # output dynamic atmospheric correction HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_DAC{INPUT_FILE.suffix}'
    else:
        # output dynamic atmospheric correction HDF5 file for NSIDC granules
        args = (PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_DAC_{2}{3}{4}_{5}_{6}_{7}_{8}_{9}.h5'
        FILENAME = file_format.format(*args)
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # full path to output file
    OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(FILENAME)

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = gz.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = gz.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()

    # read GLAH12 HDF5 file
    fileID = h5py.File(INPUT_FILE, mode='r')
    # get variables and attributes
    n_40HZ, = fileID['Data_40HZ']['Time']['i_rec_ndx'].shape
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
    ts = timescale.time.Timescale().from_deltatime(DS_UTCTime_40HZ[:],
        epoch=timescale.time._j2000_epoch, standard='UTC')
    # convert time to days relative to 1950-01-01 (MJD:33282)
    t = ts.to_deltatime(epoch=(1950,1,1,0,0,0))
    YY = np.datetime_as_string(ts.to_datetime(), unit='Y')
    # days and hours to read
    unique_hours, unique_indices = np.unique(
        [np.floor(t*24.0/6.0)*6.0, np.ceil(t*24.0/6.0)*6.0],
        return_index=True)
    days,hours = (unique_hours // 24, unique_hours % 24)
    unique_indices = unique_indices % len(t)

    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = pyTMD.spatial.datum(ellipsoid='TOPEX', units='MKS')
    wgs84 = pyTMD.spatial.datum(ellipsoid='WGS84', units='MKS')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ,elev_40HZ = pyTMD.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        topex.a_axis, topex.flat, wgs84.a_axis, wgs84.flat,
        eps=1e-12, itmax=10)

    # pyproj transformer for converting from input coordinates (EPSG)
    # to model coordinates
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_string('+proj=longlat +ellps=WGS84 +datum=WGS84 '
        '+no_defs lon_wrap=180')
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(lon_40HZ, lat_40HZ)

    # shape of DAC field
    ny,nx = (721, 1440)
    # allocate for DAC fields
    idac = np.ma.zeros((len(days), ny, nx))
    icjd = np.zeros((len(days)))
    for i,CJD in enumerate(days):
        # input file for 6-hour period
        f = f'dac_dif_{CJD:0.0f}_{hours[i]:02.0f}.nc'
        input_file = base_dir.joinpath(YY[unique_indices[i]], f)
        # check if the file exists as a compressed file
        if input_file.with_suffix('.nc.bz2').exists():
            # read bytes from compressed file
            input_file = input_file.with_suffix('.nc.bz2')
            fd = bz2.BZ2File(input_file, 'rb')
        elif input_file.exists():
            # read bytes from uncompressed file
            fd = open(input_file, 'rb')
        else:
            raise FileNotFoundError(f'File not found: {input_file}')
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
    DAC = np.ma.zeros((n_40HZ), fill_value=fv)
    DAC.data[:] = RGI.__call__(np.c_[t, iy, ix])
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
    IS_gla12_corr_attrs['lineage'] = pathlib.Path(INPUT_FILE).name
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
        FILENAME=OUTPUT_FILE,
        FILL_VALUE=IS_gla12_fill,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the correction values for ICESat data to HDF5
def HDF5_GLA12_corr_write(IS_gla12_tide, IS_gla12_attrs,
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
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version

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
        description="""Calculates and interpolates dynamic atmospheric
            corrections to ICESat/GLAS L2 GLA12 Antarctic and Greenland
            Ice Sheet elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat GLA12 file to run')
    # directory with reanalysis data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
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
        interp_DAC_ICESat_GLA12(args.directory, FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
