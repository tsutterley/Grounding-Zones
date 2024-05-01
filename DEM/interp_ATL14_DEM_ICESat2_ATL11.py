#!/usr/bin/env python
u"""
interp_ATL14_DEM_ICESat2_ATL11.py
Written by Tyler Sutterley (09/2023)
Interpolates ATL14 elevations to ICESat-2 ATL11 segment locations

COMMAND LINE OPTIONS:
    -O X, --output-directory X: input/output data directory
    -m X, --dem-model X: path to ATL14 model file
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
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/ATL11.py: reads ICESat-2 annual land ice height data files
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 09/2023: check that subsetted DEM has a valid shape and mask
        set DEM data type as float32 to reduce memory usage
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        mosaic version 3 ATL14 data for Antarctica
    Updated 07/2023: using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 11/2022: check that granule intersects ATL14 DEM
    Written 10/2022
"""
from __future__ import print_function

import re
import logging
import pathlib
import argparse
import datetime
import warnings
import collections
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
try:
    import h5py
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("h5py not available", ImportWarning)
try:
    import icesat2_toolkit as is2tk
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("icesat2_toolkit not available", ImportWarning)
try:
    import netCDF4
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("netCDF4 not available", ImportWarning)
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("pyproj not available", ImportWarning)
try:
    import timescale.time
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("timescale not available", ImportWarning)

# PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        projection_flag = 'S'
    elif GRANULE in ('03','04','05'):
        projection_flag = 'N'
    return projection_flag

# PURPOSE: read ICESat-2 annual land ice height data (ATL11)
# interpolate DEM data to x and y coordinates
def interp_ATL14_DEM_ICESat2(INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    DEM_MODEL=None,
    MODE=None):

    # log input file
    GRANULE = INPUT_FILE.name
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(GRANULE).pop()
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = is2tk.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = is2tk.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()

    # read data from input ATL11 file
    IS2_atl11_mds,IS2_atl11_attrs,IS2_atl11_pairs = \
        is2tk.io.ATL11.read_granule(INPUT_FILE, ATTRIBUTES=True)
    # get projection from ICESat-2 data file
    HEM = set_hemisphere(GRAN)
    EPSG = dict(N=3413, S=3031)
    # pyproj transformer for converting from latitude/longitude
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # read orbit info for bounding polygons
    bounding_lon = IS2_atl11_mds['orbit_info']['bounding_polygon_lon1']
    bounding_lat = IS2_atl11_mds['orbit_info']['bounding_polygon_lat1']
    # convert bounding polygon coordinates to projection
    BX, BY = transformer.transform(bounding_lon, bounding_lat)

    # verify ATL14 DEM file is iterable
    if isinstance(DEM_MODEL, str):
        DEM_MODEL = [DEM_MODEL]

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
        IMxmin = int((BX.min() - x[0])//DEM.spacing[0]) - 10
        IMxmax = int((BX.max() - x[0])//DEM.spacing[0]) + 10
        IMymin = int((BY.min() - y[0])//DEM.spacing[1]) - 10
        IMymax = int((BY.max() - y[0])//DEM.spacing[1]) + 10
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
        IMxmin = int((BX.min() - x[0])//DEM.spacing[0]) - 10
        IMxmax = int((BX.max() - x[0])//DEM.spacing[0]) + 10
        IMymin = int((BY.min() - y[0])//DEM.spacing[1]) - 10
        IMymax = int((BY.max() - y[0])//DEM.spacing[1]) + 10

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

    # use spline interpolation to calculate DEM values at coordinates
    S1 = scipy.interpolate.RectBivariateSpline(DEM.x,DEM.y,DEM.h.T,kx=1,ky=1)
    S2 = scipy.interpolate.RectBivariateSpline(DEM.x,DEM.y,DEM.h_sigma2.T,kx=1,ky=1)
    S3 = scipy.interpolate.RectBivariateSpline(DEM.x,DEM.y,DEM.ice_area.T,kx=1,ky=1)

    # copy variables for outputting to HDF5 file
    IS2_atl11_dem = {}
    IS2_atl11_fill = {}
    IS2_atl11_dims = {}
    IS2_atl11_dem_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl11_dem['ancillary_data'] = {}
    IS2_atl11_dem_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl11_dem['ancillary_data'][key] = IS2_atl11_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl11_dem_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][key].items():
            IS2_atl11_dem_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam pair within the file
    for ptx in sorted(IS2_atl11_pairs):
        # along-track (AT) reference point, latitude, longitude and time
        ref_pt = IS2_atl11_mds[ptx]['ref_pt'].copy()
        latitude = np.ma.array(IS2_atl11_mds[ptx]['latitude'],
            fill_value=IS2_atl11_attrs[ptx]['latitude']['_FillValue'])
        longitude = np.ma.array(IS2_atl11_mds[ptx]['longitude'],
            fill_value=IS2_atl11_attrs[ptx]['longitude']['_FillValue'])
        delta_time = np.ma.array(IS2_atl11_mds[ptx]['delta_time'],
            fill_value=IS2_atl11_attrs[ptx]['delta_time']['_FillValue'])

        # convert projection from latitude/longitude to DEM EPSG
        X,Y = transformer.transform(longitude, latitude)

        # check that beam pair coordinates intersect ATL14
        valid = (X >= DEM.extent[0]) & (X <= DEM.extent[1]) & \
            (Y >= DEM.extent[2]) & (Y <= DEM.extent[3])
        if not np.any(valid):
            continue

        # output data dictionaries for beam pair
        IS2_atl11_dem[ptx] = dict(ref_surf=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(ref_surf={})
        IS2_atl11_dims[ptx] = dict(ref_surf={})
        IS2_atl11_dem_attrs[ptx] = dict(ref_surf={})

        # number of average segments and number of included cycles
        # fill_value for invalid heights and corrections
        fv = IS2_atl11_attrs[ptx]['h_corr']['_FillValue']
        # shape of along-track and across-track data
        n_points,n_cycles = delta_time.shape

        # output interpolated digital elevation model
        dem_h = np.ma.zeros((n_points),fill_value=fv,dtype=np.float32)
        dem_h.mask = np.ones((n_points),dtype=bool)
        dem_h_sigma = np.ma.zeros((n_points),fill_value=fv,dtype=np.float32)
        dem_h_sigma.mask = np.ones((n_points),dtype=bool)

        # group attributes for beam
        IS2_atl11_dem_attrs[ptx]['description'] = ('Contains the primary science parameters '
            'for this data set')
        IS2_atl11_dem_attrs[ptx]['beam_pair'] = IS2_atl11_attrs[ptx]['beam_pair']
        IS2_atl11_dem_attrs[ptx]['ReferenceGroundTrack'] = IS2_atl11_attrs[ptx]['ReferenceGroundTrack']
        IS2_atl11_dem_attrs[ptx]['first_cycle'] = IS2_atl11_attrs[ptx]['first_cycle']
        IS2_atl11_dem_attrs[ptx]['last_cycle'] = IS2_atl11_attrs[ptx]['last_cycle']
        IS2_atl11_dem_attrs[ptx]['equatorial_radius'] = IS2_atl11_attrs[ptx]['equatorial_radius']
        IS2_atl11_dem_attrs[ptx]['polar_radius'] = IS2_atl11_attrs[ptx]['polar_radius']

        # geolocation, time and reference point
        # reference point
        IS2_atl11_dem[ptx]['ref_pt'] = ref_pt.copy()
        IS2_atl11_fill[ptx]['ref_pt'] = None
        IS2_atl11_dims[ptx]['ref_pt'] = None
        IS2_atl11_dem_attrs[ptx]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['ref_pt']['units'] = "1"
        IS2_atl11_dem_attrs[ptx]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_dem_attrs[ptx]['ref_pt']['long_name'] = "Reference point number"
        IS2_atl11_dem_attrs[ptx]['ref_pt']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['ref_pt']['description'] = ("The reference point is the 7 "
            "digit segment_id number corresponding to the center of the ATL06 data used for "
            "each ATL11 point.  These are sequential, starting with 1 for the first segment "
            "after an ascending equatorial crossing node.")
        IS2_atl11_dem_attrs[ptx]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number
        IS2_atl11_dem[ptx]['cycle_number'] = IS2_atl11_mds[ptx]['cycle_number'].copy()
        IS2_atl11_fill[ptx]['cycle_number'] = None
        IS2_atl11_dims[ptx]['cycle_number'] = None
        IS2_atl11_dem_attrs[ptx]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['cycle_number']['units'] = "1"
        IS2_atl11_dem_attrs[ptx]['cycle_number']['long_name'] = "Orbital cycle number"
        IS2_atl11_dem_attrs[ptx]['cycle_number']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['cycle_number']['description'] = ("Number of 91-day periods "
            "that have elapsed since ICESat-2 entered the science orbit. Each of the 1,387 "
            "reference ground track (RGTs) is targeted in the polar regions once "
            "every 91 days.")
        # delta time
        IS2_atl11_dem[ptx]['delta_time'] = delta_time.copy()
        IS2_atl11_fill[ptx]['delta_time'] = delta_time.fill_value
        IS2_atl11_dims[ptx]['delta_time'] = ['ref_pt','cycle_number']
        IS2_atl11_dem_attrs[ptx]['delta_time'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_dem_attrs[ptx]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_dem_attrs[ptx]['delta_time']['standard_name'] = "time"
        IS2_atl11_dem_attrs[ptx]['delta_time']['calendar'] = "standard"
        IS2_atl11_dem_attrs[ptx]['delta_time']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_dem_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt cycle_number latitude longitude"
        # latitude
        IS2_atl11_dem[ptx]['latitude'] = latitude.copy()
        IS2_atl11_fill[ptx]['latitude'] = latitude.fill_value
        IS2_atl11_dims[ptx]['latitude'] = ['ref_pt']
        IS2_atl11_dem_attrs[ptx]['latitude'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['latitude']['units'] = "degrees_north"
        IS2_atl11_dem_attrs[ptx]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_dem_attrs[ptx]['latitude']['long_name'] = "Latitude"
        IS2_atl11_dem_attrs[ptx]['latitude']['standard_name'] = "latitude"
        IS2_atl11_dem_attrs[ptx]['latitude']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_dem_attrs[ptx]['latitude']['valid_min'] = -90.0
        IS2_atl11_dem_attrs[ptx]['latitude']['valid_max'] = 90.0
        IS2_atl11_dem_attrs[ptx]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude
        IS2_atl11_dem[ptx]['longitude'] = longitude.copy()
        IS2_atl11_fill[ptx]['longitude'] = longitude.fill_value
        IS2_atl11_dims[ptx]['longitude'] = ['ref_pt']
        IS2_atl11_dem_attrs[ptx]['longitude'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['longitude']['units'] = "degrees_east"
        IS2_atl11_dem_attrs[ptx]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_dem_attrs[ptx]['longitude']['long_name'] = "Longitude"
        IS2_atl11_dem_attrs[ptx]['longitude']['standard_name'] = "longitude"
        IS2_atl11_dem_attrs[ptx]['longitude']['source'] = "ATL06"
        IS2_atl11_dem_attrs[ptx]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_dem_attrs[ptx]['longitude']['valid_min'] = -180.0
        IS2_atl11_dem_attrs[ptx]['longitude']['valid_max'] = 180.0
        IS2_atl11_dem_attrs[ptx]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"

        # reference surface variables
        IS2_atl11_dem_attrs[ptx]['ref_surf']['Description'] = ("The ref_surf subgroup contains "
            "parameters that describe the reference surface fit at each reference point, "
            "including slope information from ATL06, the polynomial coefficients used for the "
            "fit, and misfit statistics.")
        IS2_atl11_dem_attrs[ptx]['ref_surf']['data_rate'] = ("Data within this group "
            "are stored at the average segment rate.")

        # interpolate DEM to segment location
        dem_h.data[:] = S1.ev(X,Y)
        dem_h_sigma.data[:] = np.sqrt(S2.ev(X,Y))
        dem_ice_area = S3.ev(X,Y)
        # update masks and replace fill values
        dem_h.mask[:] = (dem_ice_area <= 0.0) | (np.abs(dem_h.data) >= 1e4)
        dem_h_sigma.mask[:] = (dem_ice_area <= 0.0) | (np.abs(dem_h.data) >= 1e4)
        dem_h.data[dem_h.mask] = dem_h.fill_value
        dem_h_sigma.data[dem_h_sigma.mask] = dem_h_sigma.fill_value

        # save ATL14 DEM elevation for pair track
        IS2_atl11_dem[ptx]['ref_surf']['dem_h'] = dem_h
        IS2_atl11_fill[ptx]['ref_surf']['dem_h'] = dem_h.fill_value
        IS2_atl11_dims[ptx]['ref_surf']['dem_h'] = ['ref_pt']
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['units'] = "meters"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['contentType'] = "referenceInformation"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['long_name'] = "DEM Height"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['description'] = ("Height of the DEM, "
            "interpolated by bivariate-spline interpolation in the DEM coordinate system "
            "to the segment location.")
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['source'] = 'ATL14'
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h']['coordinates'] = \
            "../ref_pt ../delta_time ../latitude ../longitude"

        # save ATl14 DEM elevation uncertainty for pair track
        IS2_atl11_dem[ptx]['ref_surf']['dem_h_sigma'] = dem_h_sigma
        IS2_atl11_fill[ptx]['ref_surf']['dem_h_sigma'] = dem_h_sigma.fill_value
        IS2_atl11_dims[ptx]['ref_surf']['dem_h_sigma'] = ['ref_pt']
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma'] = collections.OrderedDict()
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['units'] = "meters"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['contentType'] = "referenceInformation"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['long_name'] = "DEM Uncertainty"
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['description'] = ("Uncertainty in the "
            "DEM surface height, interpolated by bivariate-spline interpolation in the DEM "
            "coordinate system to the segment location.")
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['source'] = 'ATL14'
        IS2_atl11_dem_attrs[ptx]['ref_surf']['dem_h_sigma']['coordinates'] = \
            "../ref_pt ../delta_time ../latitude ../longitude"

    # check that there are any valid pairs in the dataset
    if bool([k for k in IS2_atl11_dem.keys() if bool(re.match(r'pt\d',k))]):
        # output HDF5 files with output DEM
        fargs = ('ATL14',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        file_format = '{0}_{1}{2}_{3}{4}_{5}_{6}{7}.h5'
        OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(file_format.format(*fargs))
        # print file information
        logging.info(f'\t{OUTPUT_FILE}')
        # write to output HDF5 file
        HDF5_ATL11_dem_write(IS2_atl11_dem, IS2_atl11_dem_attrs,
            FILENAME=OUTPUT_FILE,
            INPUT=GRANULE,
            FILL_VALUE=IS2_atl11_fill,
            DIMENSIONS=IS2_atl11_dims,
            CLOBBER=True)
        # change the permissions mode
        OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the interpolated DEM data for ICESat-2 data to HDF5
def HDF5_ATL11_dem_write(IS2_atl11_dem, IS2_atl11_attrs, INPUT=None,
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
    for k,v in IS2_atl11_dem['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = f'ancillary_data/{k}'
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam pair
    pairs = [k for k in IS2_atl11_dem.keys() if bool(re.match(r'pt\d',k))]
    for ptx in pairs:
        fileID.create_group(ptx)
        h5[ptx] = {}
        # add HDF5 group attributes for beam pair
        for att_name in ['description','beam_pair','ReferenceGroundTrack',
            'first_cycle','last_cycle','equatorial_radius','polar_radius']:
            fileID[ptx].attrs[att_name] = IS2_atl11_attrs[ptx][att_name]

        # ref_pt, cycle number, geolocation and delta_time variables
        for k in ['ref_pt','cycle_number','delta_time','latitude','longitude']:
            # values and attributes
            v = IS2_atl11_dem[ptx][k]
            attrs = IS2_atl11_attrs[ptx][k]
            fillvalue = FILL_VALUE[ptx][k]
            # Defining the HDF5 dataset variables
            val = f'{ptx}/{k}'
            if fillvalue:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[ptx][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[ptx][k]):
                    h5[ptx][k].dims[i].attach_scale(h5[ptx][dim])
            else:
                # make dimension
                h5[ptx][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[ptx][k].attrs[att_name] = att_val

        # add to output variables
        for key in ['ref_surf',]:
            fileID[ptx].create_group(key)
            h5[ptx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl11_attrs[ptx][key][att_name]
                fileID[ptx][key].attrs[att_name] = att_val
            for k,v in IS2_atl11_dem[ptx][key].items():
                # attributes
                attrs = IS2_atl11_attrs[ptx][key][k]
                fillvalue = FILL_VALUE[ptx][key][k]
                # Defining the HDF5 dataset variables
                val = f'{ptx}/{key}/{k}'
                if fillvalue:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, fillvalue=fillvalue,
                        compression='gzip')
                else:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, compression='gzip')
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[ptx][key][k]):
                    h5[ptx][key][k].dims[i].attach_scale(h5[ptx][dim])
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[ptx][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Land Ice Height'
    fileID.attrs['summary'] = ('Geophysical parameters for land ice segments '
        'needed to interpret and assess the quality of the height estimates.')
    fileID.attrs['description'] = ('Land ice parameters for each beam pair. '
        'All parameters are calculated for the same along-track increments '
        'for each beam pair and repeat.')
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
    # add attributes for input ATL11 files
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for ptx in pairs:
        lon = IS2_atl11_dem[ptx]['longitude']
        lat = IS2_atl11_dem[ptx]['latitude']
        delta_time = IS2_atl11_dem[ptx]['delta_time']
        valid = np.nonzero(delta_time != FILL_VALUE[ptx]['delta_time'])
        # setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time[valid].min() if (delta_time[valid].min() < tmn) else tmn
        tmx = delta_time[valid].max() if (delta_time[valid].max() > tmx) else tmx
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
        description="""Interpolate DEMs to ICESat-2 ATL11 annual land
            ice height locations
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('file',
        type=pathlib.Path,
        help='ICESat-2 ATL11 file to run')
    # directory with output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # full path to ATL14 digital elevation file
    parser.add_argument('--dem-model','-m',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL14 DEM file to run')
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

    # create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # run program with parameters
    interp_ATL14_DEM_ICESat2(args.file,
        OUTPUT_DIRECTORY=args.output_directory,
        DEM_MODEL=args.dem_model,
        MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
