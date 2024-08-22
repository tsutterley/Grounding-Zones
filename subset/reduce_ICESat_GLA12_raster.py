#!/usr/bin/env python
u"""
reduce_ICESat_GLA12_raster.py
Written by Tyler Sutterley (08/2024)

Create masks for reducing ICESat/GLAS L2 GLA12 Antarctic and Greenland
    Ice Sheet elevation data data using raster imagery

COMMAND LINE OPTIONS:
    -R X, --raster X: Input raster file
    -F X, --format X: Input raster file format
        netCDF4
        HDF5
        GTiff
        cog
    -v X, --variables X: variable names of data in HDF5 or netCDF4 file
        x, y and data variable names
    -P X, --projection X: spatial projection as EPSG code or PROJ4 string
        4326: latitude and longitude coordinates on WGS84 reference ellipsoid
    -S X, --sigma X: Standard deviation for Gaussian kernel
    -O X, --output X: Output mask file name
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
        https://pypi.python.org/pypi/GDAL/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/ATL06.py: reads ICESat-2 land ice along-track height data files
    spatial.py: utilities for reading and writing spatial data
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 08/2024: changed from 'geotiff' to 'GTiff' and 'cog' formats
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 04/2024: use timescale for temporal operations
    Updated 03/2024: use pathlib to define and operate on paths
    Updated 12/2022: single implicit import of altimetry tools
        refactored ICESat-2 data product read programs under io
    Updated 06/2022: added option sigma to Gaussian filter raster images
    Updated 05/2022: use argparse descriptions within sphinx documentation
    Written 11/2021
"""
from __future__ import print_function

import re
import logging
import pathlib
import argparse
import numpy as np
import scipy.ndimage
import scipy.spatial
import scipy.interpolate
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
pyTMD = gz.utilities.import_dependency('pyTMD')
pyproj = gz.utilities.import_dependency('pyproj')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: try to get the projection information for the input file
def get_projection(attributes, PROJECTION):
    # coordinate reference system string from file
    try:
        crs = pyproj.CRS.from_string(attributes['projection'])
    except (ValueError,pyproj.exceptions.CRSError):
        pass
    else:
        return crs
    # EPSG projection code
    try:
        crs = pyproj.CRS.from_epsg(int(PROJECTION))
    except (ValueError,pyproj.exceptions.CRSError):
        pass
    else:
        return crs
    # coordinate reference system string
    try:
        crs = pyproj.CRS.from_string(PROJECTION)
    except (ValueError,pyproj.exceptions.CRSError):
        pass
    else:
        return crs
    # no projection can be made
    raise pyproj.exceptions.CRSError

# PURPOSE: find a valid Delaunay triangulation for coordinates x0 and y0
# http://www.qhull.org/html/qhull.htm#options
# Attempt 1: standard qhull options Qt Qbb Qc Qz
# Attempt 2: rescale and center the inputs with option QbB
# Attempt 3: joggle the inputs to find a triangulation with option QJ
# if no passing triangulations: exit with empty list
def find_valid_triangulation(x0, y0, max_points=1e6):
    """
    Attempt to find a valid Delaunay triangulation for coordinates

    - Attempt 1: ``Qt Qbb Qc Qz``
    - Attempt 2: ``Qt Qc QbB``
    - Attempt 3: ``QJ QbB``

    Parameters
    ----------
    x0: float
        x-coordinates
    y0: float
        y-coordinates
    max_points: int or float, default 1e6
        Maximum number of coordinates to attempt to triangulate
    """
    # don't attempt triangulation if there are a large number of points
    if (len(x0) > max_points):
        # if too many points: set triangle as an empty list
        logging.info('Too many points for triangulation')
        return (None,[])

    # Attempt 1: try with standard options Qt Qbb Qc Qz
    # Qt: triangulated output, all facets will be simplicial
    # Qbb: scale last coordinate to [0,m] for Delaunay triangulations
    # Qc: keep coplanar points with nearest facet
    # Qz: add point-at-infinity to Delaunay triangulation

    # Attempt 2 in case of qhull error from Attempt 1 try Qt Qc QbB
    # Qt: triangulated output, all facets will be simplicial
    # Qc: keep coplanar points with nearest facet
    # QbB: scale input to unit cube centered at the origin

    # Attempt 3 in case of qhull error from Attempt 2 try QJ QbB
    # QJ: joggle input instead of merging facets
    # QbB: scale input to unit cube centered at the origin

    # try each set of qhull_options
    points = np.concatenate((x0[:,None],y0[:,None]),axis=1)
    for i,opt in enumerate(['Qt Qbb Qc Qz','Qt Qc QbB','QJ QbB']):
        logging.info(f'qhull option: {opt}')
        try:
            triangle = scipy.spatial.Delaunay(points.data, qhull_options=opt)
        except scipy.spatial.qhull.QhullError:
            pass
        else:
            return (i+1,triangle)

    # if still errors: set triangle as an empty list
    return (None,[])

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12)
# reduce to a masked region using raster imagery
def reduce_ICESat_GLA12_raster(INPUT_FILE,
    MASK=None,
    FORMAT=None,
    VARIABLES=[],
    OUTPUT=None,
    PROJECTION=None,
    SIGMA=0.0,
    TOLERANCE=0.5,
    VERBOSE=False,
    MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)
    # input granule basename
    GRANULE = INPUT_FILE.name

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
    VAR = 'MASK'
    try:
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = \
            rx.findall(GRANULE).pop()
    except (ValueError, IndexError):
        # output mask HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_{VAR}{INPUT_FILE.suffix}'
    else:
        # output mask HDF5 file for NSIDC granules
        fargs = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_{2}_{3}{4}{5}_{6}_{7}_{8}_{9}_{10}.h5'
        FILENAME = file_format.format(*fargs)

    # default output from input file
    if OUTPUT is None:
        OUTPUT = INPUT_FILE.with_name(FILENAME)

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = gz.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = gz.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()

    # Open the HDF5 file for reading
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

    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = pyTMD.datum(ellipsoid='TOPEX', units='MKS')
    wgs84 = pyTMD.datum(ellipsoid='WGS84', units='MKS')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ, elev_40HZ = pyTMD.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        topex.a_axis, topex.flat, wgs84.a_axis, wgs84.flat,
        eps=1e-12, itmax=10)

    # read raster image for spatial coordinates and data
    MASK = pathlib.Path(MASK).expanduser().absolute()
    dinput = pyTMD.spatial.from_file(MASK, FORMAT,
        xname=VARIABLES[0], yname=VARIABLES[1], varname=VARIABLES[2])
    # raster extents
    xmin,xmax,ymin,ymax = np.copy(dinput['attributes']['extent'])
    # check that x and y are strictly increasing
    if (np.sign(dinput['attributes']['spacing'][0]) == -1):
        dinput['x'] = dinput['x'][::-1]
        dinput['data'] = dinput['data'][:,::-1]
    if (np.sign(dinput['attributes']['spacing'][1]) == -1):
        dinput['y'] = dinput['y'][::-1]
        dinput['data'] = dinput['data'][::-1,:]
    # find valid points within mask
    indy,indx = np.nonzero(dinput['data'])
    # check that input points are within convex hull of valid model points
    gridx,gridy = np.meshgrid(dinput['x'],dinput['y'])
    v,triangle = find_valid_triangulation(gridx[indy,indx],gridy[indy,indx])
    # gaussian filter mask to increase coverage
    if (SIGMA > 0):
        # convert nan values to 0
        dinput['data'] = np.nan_to_num(dinput['data'], nan=0.0)
        ii,jj = np.nonzero(np.logical_not(dinput['data'].mask) &
            (dinput['data'] != 0.0))
        # gaussian filter image
        dinput['data'] = scipy.ndimage.gaussian_filter(dinput['data'],
            SIGMA, mode='constant', cval=0)
        # return original mask values to true
        dinput['data'][ii,jj] = 1.0
    # create an interpolator for input raster data
    logging.info('Building Spline Interpolator')
    SPL = scipy.interpolate.RectBivariateSpline(dinput['x'], dinput['y'],
        dinput['data'].T, kx=1, ky=1)

    # convert projection from input coordinates (EPSG) to data coordinates
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = get_projection(dinput['attributes'], PROJECTION)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    logging.info(crs2.to_proj4())

    # convert latitude/longitude to raster image projection
    X,Y = transformer.transform(lon_40HZ, lat_40HZ)

    # check where points are within complex hull of triangulation
    # or within the bounds of the input raster image
    if v:
        interp_points = np.concatenate((X[:,None],Y[:,None]),axis=1)
        valid = (triangle.find_simplex(interp_points) >= 0)
    else:
        valid = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)

    # interpolate raster mask to points
    interp_mask = np.zeros((n_40HZ),dtype=bool)
    # skip interpolation if no data within bounds of raster image
    if np.any(valid):
        interp_mask[valid] = (SPL.ev(X[valid], Y[valid]) >= TOLERANCE)
    else:
        return

    # copy variables for outputting to HDF5 file
    IS_gla12_mask = dict(Data_40HZ={})
    IS_gla12_mask_attrs = dict(Data_40HZ={})

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
        IS_gla12_mask_attrs[att] = fileID.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_mask_attrs['Campaign'] = fileID['ANCILLARY_DATA'].attrs['Campaign']

    # add attributes for input GLA12 file
    IS_gla12_mask_attrs['lineage'] = pathlib.Path(INPUT_FILE).name
    # update geospatial ranges for ellipsoid
    IS_gla12_mask_attrs['geospatial_lat_min'] = np.min(lat_40HZ)
    IS_gla12_mask_attrs['geospatial_lat_max'] = np.max(lat_40HZ)
    IS_gla12_mask_attrs['geospatial_lon_min'] = np.min(lon_40HZ)
    IS_gla12_mask_attrs['geospatial_lon_max'] = np.max(lon_40HZ)
    IS_gla12_mask_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_mask_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_mask_attrs['geospatial_ellipsoid'] = "WGS84"

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_mask_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time, geolocation and geophysical groups
    for var in ['Time','Geolocation','Geophysical']:
        IS_gla12_mask['Data_40HZ'][var] = {}
        IS_gla12_mask_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_mask_attrs['Data_40HZ'][var][att_name] = att_val

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_mask_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time and geolocation groups
    for var in ['Time','Geolocation']:
        IS_gla12_mask['Data_40HZ'][var] = {}
        IS_gla12_mask_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_mask_attrs['Data_40HZ'][var][att_name] = att_val

    # J2000 time
    IS_gla12_mask['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_mask_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in fileID['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_mask_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    # record
    IS_gla12_mask['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_mask_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Time']['i_rec_ndx']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_mask_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    # latitude
    IS_gla12_mask['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lat']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    # longitude
    IS_gla12_mask['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lon']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    # subsetting variables
    IS_gla12_mask['Data_40HZ']['Subsetting'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['Description'] = \
        ("The subsetting group contains parameters used to reduce values "
        "to specific regions of interest.")

    # output mask
    IS_gla12_mask['Data_40HZ']['Subsetting']['d_mask'] = interp_mask
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_mask'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_mask']['contentType'] = \
        "referenceInformation"
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_mask']['long_name'] = \
        'Mask'
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_mask']['description'] = \
        'Mask calculated using raster image'
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_mask']['source'] = MASK.name
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_mask']['sigma'] = SIGMA
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_mask']['tolerance'] = TOLERANCE
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_mask']['coordinates'] = \
        "../DS_UTCTime_40"

    # print file information
    logging.info(f'\t{str(OUTPUT)}')
    # write to output HDF5 file
    HDF5_GLA12_mask_write(IS_gla12_mask, IS_gla12_mask_attrs,
        FILENAME=OUTPUT, CLOBBER=True)
    # change the permissions mode
    OUTPUT.chmod(mode=MODE)
    # close the input file
    fileID.close()

# PURPOSE: outputting the mask values for ICESat data to HDF5
def HDF5_GLA12_mask_write(IS_gla12_mask, IS_gla12_attrs,
    FILENAME='', CLOBBER=False):
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
    val = IS_gla12_mask['Data_40HZ']['DS_UTCTime_40']
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
    for group in ['Time','Geolocation','Subsetting']:
        # add group to dict
        h5['Data_40HZ'][group] = {}
        # create Data_40HZ group
        fileID.create_group(f'Data_40HZ/{group}')
        # add HDF5 group attributes
        for att_name,att_val in IS_gla12_attrs['Data_40HZ'][group].items():
            if not isinstance(att_val,dict):
                fileID['Data_40HZ'][group].attrs[att_name] = att_val
        # for each variable in the group
        for key,val in IS_gla12_mask['Data_40HZ'][group].items():
            attrs = IS_gla12_attrs['Data_40HZ'][group][key]
            # Defining the HDF5 dataset variables
            var = f'Data_40HZ/{group}/{key}'
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
        description="""Create masks for reducing ICESat/GLAS L2
            GLA12 Antarctic and Greenland Ice Sheet elevation data
            using raster imagery
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('file',
        type=pathlib.Path,
        help='ICESat GLA12 file to run')
    # use default output file name
    parser.add_argument('--output','-O',
        type=pathlib.Path,
        help='Name and path of output file')
    # input raster file and file format
    parser.add_argument('--raster','-R',
        type=pathlib.Path,
        help='Input raster file')
    parser.add_argument('--format','-F',
        type=str, default='GTiff',
        choices=('netCDF4','HDF5','GTiff','cog'),
        help='Input raster file format')
    # variable names of data in HDF5 or netCDF4 file
    parser.add_argument('--variables','-v',
        type=str, nargs='+', default=['x','y','data'],
        help='Variable names of data in HDF5 or netCDF4 files')
    # spatial projection (EPSG code or PROJ4 string)
    parser.add_argument('--projection','-P',
        type=str, default='4326',
        help='Spatial projection as EPSG code or PROJ4 string')
    # Gaussian filter raster image to increase coverage
    parser.add_argument('--sigma','-S',
        type=float, default=0.0,
        help='Standard deviation for Gaussian kernel')
    # tolerance in interpolated mask to set as valid
    parser.add_argument('--tolerance','-T',
        type=float, default=0.5,
        help='Tolerance to set as valid mask')
    # verbosity settings
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permissions mode of output files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # run raster mask program with parameters
    reduce_ICESat_GLA12_raster(args.file,
        MASK=args.raster,
        FORMAT=args.format,
        VARIABLES=args.variables,
        PROJECTION=args.projection,
        SIGMA=args.sigma,
        TOLERANCE=args.tolerance,
        OUTPUT=args.output,
        VERBOSE=args.verbose,
        MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
