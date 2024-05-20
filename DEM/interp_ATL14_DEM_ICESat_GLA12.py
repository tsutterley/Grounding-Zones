#!/usr/bin/env python
u"""
interp_ATL14_DEM_ICESat_GLA12.py
Written by Tyler Sutterley (05/2024)
Interpolates ATL14 elevations to locations of ICESat/GLAS L2
    GLA12 Antarctic and Greenland Ice Sheet elevation data

COMMAND LINE OPTIONS:
    -O X, --output-directory X: input/output data directory
    -m X, --model X: path to ATL14 model file
    -H X, --hemisphere X: Region of interest to run
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Written 05/2024
"""
from __future__ import print_function

import re
import logging
import pathlib
import argparse
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
netCDF4 = gz.utilities.import_dependency('netCDF4')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

def read_ATL14_model(DEM_MODEL, BOUNDS=[-np.inf, np.inf, -np.inf, np.inf]):
    """
    Read ATL14 DEM model files within spatial bounds

    Parameters
    ----------
    DEM_MODEL: list of ATL14 DEM model files
    BOUNDS: spatial bounds to crop DEM model files
    """
    # subset ATL14 elevation field to bounds
    DEM = gz.mosaic()
    # iterate over each ATL14 DEM file
    for MODEL in DEM_MODEL:
        # check if DEM is an s3 presigned url
        if str(MODEL).startswith('s3:'):
            is2tk.utilities.attempt_login('urs.earthdata.nasa.gov',
                authorization_header=True)
            session = is2tk.utilities.s3_filesystem()
            MODEL = session.open(MODEL, mode='rb')
        else:
            MODEL = pathlib.Path(MODEL).expanduser().absolute()

        # open ATL14 DEM file for reading
        logging.info(str(MODEL))
        with netCDF4.Dataset(MODEL, mode='r') as fileID:
            # get original grid coordinates
            x = fileID.variables['x'][:].copy()
            y = fileID.variables['y'][:].copy()
            # fill_value for invalid heights
            fv = fileID['h'].getncattr('_FillValue')
        # update the mosaic grid spacing
        DEM.update_spacing(x, y)
        # get size of DEM
        ny, nx = len(y), len(x)

        # determine buffered bounds of data in image coordinates
        # (affine transform)
        IMxmin = int((BOUNDS[0] - x[0])//DEM.spacing[0]) - 10
        IMxmax = int((BOUNDS[1] - x[0])//DEM.spacing[0]) + 10
        IMymin = int((BOUNDS[2] - y[0])//DEM.spacing[1]) - 10
        IMymax = int((BOUNDS[3] - y[0])//DEM.spacing[1]) + 10
        # get buffered bounds of data
        # and convert invalid values to 0
        indx = slice(np.maximum(IMxmin,0), np.minimum(IMxmax,nx), 1)
        indy = slice(np.maximum(IMymin,0), np.minimum(IMymax,ny), 1)
        DEM.update_bounds(x[indx], y[indy])

    # check that DEM has a valid shape
    if np.any(np.sign(DEM.shape) == -1):
        raise ValueError('Values outside of ATL14 range')

    # fill ATL14 to mosaic
    DEM.h = np.ma.zeros(DEM.shape, dtype=np.float32, fill_value=fv)
    DEM.h_sigma2 = np.ma.zeros(DEM.shape, dtype=np.float32, fill_value=fv)
    DEM.ice_area = np.ma.zeros(DEM.shape, dtype=np.float32, fill_value=fv)
    # iterate over each ATL14 DEM file
    for MODEL in DEM_MODEL:
        # check if DEM is an s3 presigned url
        if str(MODEL).startswith('s3:'):
            is2tk.utilities.attempt_login('urs.earthdata.nasa.gov',
                authorization_header=True)
            session = is2tk.utilities.s3_filesystem()
            MODEL = session.open(MODEL, mode='rb')
        else:
            MODEL = pathlib.Path(MODEL).expanduser().absolute()

        # open ATL14 DEM file for reading
        fileID = netCDF4.Dataset(MODEL, mode='r')
        # get original grid coordinates
        x = fileID.variables['x'][:].copy()
        y = fileID.variables['y'][:].copy()
        # get size of DEM
        ny, nx = len(y), len(x)

        # determine buffered bounds of data in image coordinates
        # (affine transform)
        IMxmin = int((BOUNDS[0] - x[0])//DEM.spacing[0]) - 10
        IMxmax = int((BOUNDS[1] - x[0])//DEM.spacing[0]) + 10
        IMymin = int((BOUNDS[2] - y[0])//DEM.spacing[1]) - 10
        IMymax = int((BOUNDS[3] - y[0])//DEM.spacing[1]) + 10

        # get buffered bounds of data
        # and convert invalid values to 0
        indx = slice(np.maximum(IMxmin,0), np.minimum(IMxmax,nx), 1)
        indy = slice(np.maximum(IMymin,0), np.minimum(IMymax,ny), 1)
        # get the image coordinates of the input file
        iy, ix = DEM.image_coordinates(x[indx], y[indy])
        # create mosaic of DEM variables
        if np.any(iy) and np.any(ix):
            DEM.h[iy, ix] = fileID['h'][indy, indx]
            DEM.h_sigma2[iy, ix] = fileID['h_sigma'][indy, indx]**2
            DEM.ice_area[iy, ix] = fileID['ice_area'][indy, indx]
        # close the ATL14 file
        fileID.close()

    # update masks for DEM
    for key in ['h', 'h_sigma2', 'ice_area']:
        val = getattr(DEM, key)
        val.mask = (val.data == val.fill_value) | np.isnan(val.data)
        val.data[val.mask] = val.fill_value

    # return the DEM object
    return DEM

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12) from NSIDC
# interpolate DEM data to x and y coordinates
def interp_ATL14_DEM_ICESat(INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    DEM_MODEL=None,
    HEM=None,
    MODE=None):

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # ATL14 model for the hemisphere
    REGION = dict(N='GL', S='AA')
    MODEL = f'ATL14_{REGION[HEM]}'
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
        # output inverse-barometer response HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_{MODEL}_{INPUT_FILE.suffix}'
    else:
        # output inverse-barometer response HDF5 file for NSIDC granules
        args = (PRD,RL,MODEL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_{2}_{3}{4}{5}_{6}_{7}_{8}_{9}_{10}.h5'
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
    rec_ndx_1HZ = fileID['Data_1HZ']['Time']['i_rec_ndx'][:].copy()
    rec_ndx_40HZ = fileID['Data_40HZ']['Time']['i_rec_ndx'][:].copy()
    # seconds since 2000-01-01 12:00:00 UTC (J2000)
    DS_UTCTime_40HZ = fileID['Data_40HZ']['DS_UTCTime_40'][:].copy()
    # ICESat track number
    i_track_1HZ = fileID['Data_1HZ']['Geolocation']['i_track'][:].copy()
    i_track_40HZ = np.zeros((n_40HZ), dtype=i_track_1HZ.dtype)
    # Latitude (degrees North)
    lat_TPX = fileID['Data_40HZ']['Geolocation']['d_lat'][:].copy()
    # Longitude (degrees East)
    lon_40HZ = fileID['Data_40HZ']['Geolocation']['d_lon'][:].copy()
    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
    elev_TPX = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'][:].copy()
    fv = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'].attrs['_FillValue']
    # map 1HZ data to 40HZ data
    for k,record in enumerate(rec_ndx_1HZ):
        # indice mapping the 40HZ data to the 1HZ data
        map_1HZ_40HZ, = np.nonzero(rec_ndx_40HZ == record)
        i_track_40HZ[map_1HZ_40HZ] = i_track_1HZ[k]

    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = pyTMD.datum(ellipsoid='TOPEX', units='MKS')
    wgs84 = pyTMD.datum(ellipsoid='WGS84', units='MKS')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ,elev_40HZ = is2tk.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        topex.a_axis, topex.flat, wgs84.a_axis, wgs84.flat, eps=1e-12, itmax=10)

    # pyproj transformer for converting from latitude/longitude
    EPSG = dict(N=3413, S=3031)
    SIGN = dict(N=1.0, S=-1.0)
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # convert from latitude/longitude to polar stereographic
    X,Y = transformer.transform(lon_40HZ, lat_40HZ)

    # verify ATL14 DEM file is iterable
    if isinstance(DEM_MODEL, str):
        DEM_MODEL = [DEM_MODEL]

    # output interpolated digital elevation model
    dem_h = np.ma.zeros((n_40HZ), fill_value=fv, dtype=np.float32)
    dem_h.mask = np.ma.zeros((n_40HZ), dtype=bool)
    dem_h_sigma = np.ma.zeros((n_40HZ), fill_value=fv, dtype=np.float32)
    dem_h_sigma.mask = np.ma.zeros((n_40HZ), dtype=bool)
    dem_ice_area = np.zeros((n_40HZ))
    # iterate over reference tracks as the individual ICESat files are
    # too spatially extensive to interpolate all ATL14 points at once
    for i, track in enumerate(np.unique(i_track_40HZ)):
        # extract GLA12 data for track
        valid, = np.nonzero((i_track_40HZ == track) & 
            (np.sign(lat_40HZ) == SIGN[HEM]) & (elev_TPX != fv))
        # check if there are valid points for the track
        if not valid.any():
            continue
        # get bounds of valid points within hemisphere
        xmin, xmax = np.min(X[valid]), np.max(X[valid])
        ymin, ymax = np.min(Y[valid]), np.max(Y[valid])
        # read ATL14 model for the hemisphere
        DEM = read_ATL14_model(DEM_MODEL, BOUNDS=[xmin, xmax, ymin, ymax])
        # create 2D interpolation of DEM data
        S1 = scipy.interpolate.RectBivariateSpline(DEM.y, DEM.x, DEM.h.data)
        S2 = scipy.interpolate.RectBivariateSpline(DEM.y, DEM.x, DEM.h_sigma2.data)
        S3 = scipy.interpolate.RectBivariateSpline(DEM.y, DEM.x, DEM.ice_area.data)
        # interpolate DEM to GLA12 locations
        dem_h.data[valid] = S1.ev(X[valid], Y[valid])
        dem_h_sigma.data[valid] = np.sqrt(S2.ev(X[valid], Y[valid]))
        dem_ice_area[valid] = S3.ev(X[valid], Y[valid])
        # clear DEM variable
        DEM = None

    # update masks and replace fill values
    dem_h.mask[:] = (dem_ice_area <= 0.0) | (np.abs(dem_h.data) >= 1e4)
    dem_h_sigma.mask[:] = (dem_ice_area <= 0.0) | (np.abs(dem_h.data) >= 1e4)
    dem_h.data[dem_h.mask] = dem_h.fill_value
    dem_h_sigma.data[dem_h_sigma.mask] = dem_h_sigma.fill_value

    # copy variables for outputting to HDF5 file
    IS_gla12_dem = dict(Data_40HZ={})
    IS_gla12_fill = dict(Data_40HZ={})
    IS_gla12_dem_attrs = dict(Data_40HZ={})

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
        IS_gla12_dem_attrs[att] = fileID.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_dem_attrs['Campaign'] = fileID['ANCILLARY_DATA'].attrs['Campaign']

    # add attributes for input GLA12 file
    IS_gla12_dem_attrs['lineage'] = pathlib.Path(INPUT_FILE).name
    # update geospatial ranges for ellipsoid
    IS_gla12_dem_attrs['geospatial_lat_min'] = np.min(lat_40HZ)
    IS_gla12_dem_attrs['geospatial_lat_max'] = np.max(lat_40HZ)
    IS_gla12_dem_attrs['geospatial_lon_min'] = np.min(lon_40HZ)
    IS_gla12_dem_attrs['geospatial_lon_max'] = np.max(lon_40HZ)
    IS_gla12_dem_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_dem_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_dem_attrs['geospatial_ellipsoid'] = "WGS84"

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_dem_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time, geolocation and geophysical groups
    for var in ['Time','Geolocation','Geophysical']:
        IS_gla12_dem['Data_40HZ'][var] = {}
        IS_gla12_fill['Data_40HZ'][var] = {}
        IS_gla12_dem_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_dem_attrs['Data_40HZ'][var][att_name] = att_val

    # J2000 time
    IS_gla12_dem['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_fill['Data_40HZ']['DS_UTCTime_40'] = None
    IS_gla12_dem_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in fileID['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_dem_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    # record
    IS_gla12_dem['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_fill['Data_40HZ']['Time']['i_rec_ndx'] = None
    IS_gla12_dem_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_dem_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    # latitude
    IS_gla12_dem['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lat'] = None
    IS_gla12_dem_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_dem_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    # longitude
    IS_gla12_dem['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lon'] = None
    IS_gla12_dem_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_dem_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    # ATL14 DEM elevation
    IS_gla12_dem['Data_40HZ']['Geophysical']['d_DEM_elv'] = dem_h.copy()
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_DEM_elv'] = dem_h.fill_value
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv'] = {}
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['units'] = "meters"
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['long_name'] = \
        "DEM Height"
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['description'] = \
        ("Height of the DEM, interpolated by bivariate-spline interpolation in the DEM "
        "coordinate system to the segment location.")
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['source'] = 'ATL14'
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['coordinates'] = \
        "../DS_UTCTime_40"

    # ATL14 DEM elevation uncertainty
    IS_gla12_dem['Data_40HZ']['Geophysical']['d_DEM_sigma'] = dem_h_sigma.copy()
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_DEM_sigma'] = dem_h_sigma.fill_value
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_sigma'] = {}
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_sigma']['units'] = "meters"
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_sigma']['long_name'] = \
        "DEM Uncertainty"
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_sigma']['description'] = \
        ("Uncertainty in the DEM surface height, interpolated by bivariate-spline "
        "interpolation in the DEM coordinate system to the segment location.")
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_sigma']['source'] = 'ATL14'
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_sigma']['coordinates'] = \
        "../DS_UTCTime_40"

    # close the input HDF5 file
    fileID.close()

    # print file information
    logging.info(f'\t{OUTPUT_FILE}')
    HDF5_GLA12_dem_write(IS_gla12_dem, IS_gla12_dem_attrs,
        FILENAME=OUTPUT_FILE,
        FILL_VALUE=IS_gla12_fill,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the DEM values for ICESat data to HDF5
def HDF5_GLA12_dem_write(IS_gla12_tide, IS_gla12_attrs,
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
        description="""Interpolate DEMs to ICESat/GLAS L2 GLA12 Antarctic
            and Greenland Ice Sheet elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('file',
        type=pathlib.Path,
        help='ICESat GLA12 file to run')
    # directory with output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # full path to ATL14 digital elevation file
    parser.add_argument('--dem-model','-m',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL14 DEM file to run')
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='N', choices=('N','S'),
        help='Hemisphere')
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

    # create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # run program with parameters
    interp_ATL14_DEM_ICESat(args.file,
        OUTPUT_DIRECTORY=args.output_directory,
        DEM_MODEL=args.dem_model,
        HEM=args.hemisphere,
        MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
