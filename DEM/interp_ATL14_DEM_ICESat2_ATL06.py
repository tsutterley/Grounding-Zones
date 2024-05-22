#!/usr/bin/env python
u"""
interp_ATL14_DEM_ICESat2_ATL06.py
Written by Tyler Sutterley (05/2024)
Interpolates ATL14 elevations to locations of ICESat-2 ATL06 segments

COMMAND LINE OPTIONS:
    -O X, --output-directory X: input/output data directory
    -m X, --model X: path to ATL14 model file
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
    io/ATL06.py: reads ICESat-2 land ice along-track height data files
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
        use regular grid interpolator for DEM data instead of spline
        refactor read ATL14 function to external module 
    Updated 09/2023: check that subsetted DEM has a valid shape and mask
        set DEM data type as float32 to reduce memory usage
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        mosaic version 3 ATL14 data for Antarctica
    Updated 07/2023: using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 11/2022: check that granule intersects ATL14 DEM
    Written 11/2022
"""
from __future__ import print_function

import re
import logging
import pathlib
import argparse
import datetime
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
netCDF4 = gz.utilities.import_dependency('netCDF4')
pyproj = gz.utilities.import_dependency('pyproj')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        projection_flag = 'S'
    elif GRANULE in ('03','04','05'):
        projection_flag = 'N'
    return projection_flag

# PURPOSE: read ICESat-2 land ice height data (ATL06)
# interpolate DEM data to x and y coordinates
def interp_ATL14_DEM_ICESat2(INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    DEM_MODEL=None,
    MODE=None):

    # log input file
    GRANULE = INPUT_FILE.name
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS,AUX = \
        rx.findall(GRANULE).pop()
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

    # read data from input ATL06 file
    IS2_atl06_mds,IS2_atl06_attrs,IS2_atl06_beams = \
        is2tk.io.ATL06.read_granule(INPUT_FILE, ATTRIBUTES=True)
    # get projection from ICESat-2 data file
    HEM = set_hemisphere(GRN)
    EPSG = dict(N=3413, S=3031)
    # pyproj transformer for converting from latitude/longitude
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # read orbit info for bounding polygons
    bounding_lon = IS2_atl06_mds['orbit_info']['bounding_polygon_lon1']
    bounding_lat = IS2_atl06_mds['orbit_info']['bounding_polygon_lat1']
    # convert bounding polygon coordinates to projection
    BX, BY = transformer.transform(bounding_lon, bounding_lat)
    BOUNDS = [BX.min(), BX.max(), BY.min(), BY.max()]

    # read ATL14 DEM model files within spatial bounds
    DEM = gz.io.ATL14(DEM_MODEL, BOUNDS=BOUNDS)
    # verify coordinate reference systems match
    assert pyproj.CRS.is_exact_same(crs2, DEM.crs), \
        'Inconsistent coordinate reference systems'
    # create 2D interpolation of DEM data
    R1 = scipy.interpolate.RegularGridInterpolator((DEM.y, DEM.x),
        DEM.h, bounds_error=False)
    R2 = scipy.interpolate.RegularGridInterpolator((DEM.y, DEM.x),
        DEM.h_sigma**2, bounds_error=False)
    R3 = scipy.interpolate.RegularGridInterpolator((DEM.y, DEM.x),
        DEM.ice_area, bounds_error=False)
    # clear DEM variable
    DEM = None

    # copy variables for outputting to HDF5 file
    IS2_atl06_dem = {}
    IS2_atl06_fill = {}
    IS2_atl06_dims = {}
    IS2_atl06_dem_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl06_dem['ancillary_data'] = {}
    IS2_atl06_dem_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl06_dem['ancillary_data'][key] = IS2_atl06_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl06_dem_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][key].items():
            IS2_atl06_dem_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl06_beams):
        # variables for beam
        val = IS2_atl06_mds[gtx]['land_ice_segments']
        # number of segments
        n_seg, = val['segment_id'].shape
        # fill_value for invalid heights and corrections
        fv = IS2_atl06_attrs[gtx]['land_ice_segments']['h_li']['_FillValue']

        # convert projection from latitude/longitude to DEM EPSG
        X,Y = transformer.transform(val['longitude'], val['latitude'])

        # check that beam coordinates intersect ATL14
        valid = (X >= DEM.extent[0]) & (X <= DEM.extent[1]) & \
            (Y >= DEM.extent[2]) & (Y <= DEM.extent[3])
        if not np.any(valid):
            continue

        # output data dictionaries for beam
        IS2_atl06_dem[gtx] = dict(land_ice_segments={})
        IS2_atl06_fill[gtx] = dict(land_ice_segments={})
        IS2_atl06_dims[gtx] = dict(land_ice_segments={})
        IS2_atl06_dem_attrs[gtx] = dict(land_ice_segments={})

        # output interpolated digital elevation model
        dem_h = np.ma.zeros((n_seg),fill_value=fv,dtype=np.float32)
        dem_h.mask = np.ones((n_seg),dtype=bool)
        dem_h_sigma = np.ma.zeros((n_seg),fill_value=fv,dtype=np.float32)
        dem_h_sigma.mask = np.ones((n_seg),dtype=bool)

        # group attributes for beam
        IS2_atl06_dem_attrs[gtx]['Description'] = IS2_atl06_attrs[gtx]['Description']
        IS2_atl06_dem_attrs[gtx]['atlas_pce'] = IS2_atl06_attrs[gtx]['atlas_pce']
        IS2_atl06_dem_attrs[gtx]['atlas_beam_type'] = IS2_atl06_attrs[gtx]['atlas_beam_type']
        IS2_atl06_dem_attrs[gtx]['groundtrack_id'] = IS2_atl06_attrs[gtx]['groundtrack_id']
        IS2_atl06_dem_attrs[gtx]['atmosphere_profile'] = IS2_atl06_attrs[gtx]['atmosphere_profile']
        IS2_atl06_dem_attrs[gtx]['atlas_spot_number'] = IS2_atl06_attrs[gtx]['atlas_spot_number']
        IS2_atl06_dem_attrs[gtx]['sc_orientation'] = IS2_atl06_attrs[gtx]['sc_orientation']
        # group attributes for land_ice_segments
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['Description'] = ("The land_ice_segments group "
            "contains the primary set of derived products. This includes geolocation, height, and "
            "standard error and quality measures for each segment. This group is sparse, meaning "
            "that parameters are provided only for pairs of segments for which at least one beam "
            "has a valid surface-height measurement.")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['data_rate'] = ("Data within this group are "
            "sparse.  Data values are provided only for those ICESat-2 20m segments where at "
            "least one beam has a valid land ice height measurement.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl06_dem[gtx]['land_ice_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['delta_time'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['delta_time'] = None
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['standard_name'] = "time"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['calendar'] = "standard"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['delta_time']['coordinates'] = \
            "segment_id latitude longitude"
        # latitude
        IS2_atl06_dem[gtx]['land_ice_segments']['latitude'] = val['latitude'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['latitude'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['latitude'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['units'] = "degrees_north"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['valid_min'] = -90.0
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['valid_max'] = 90.0
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['latitude']['coordinates'] = \
            "segment_id delta_time longitude"
        # longitude
        IS2_atl06_dem[gtx]['land_ice_segments']['longitude'] = val['longitude'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['longitude'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['longitude'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['units'] = "degrees_east"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['valid_min'] = -180.0
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['valid_max'] = 180.0
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['longitude']['coordinates'] = \
            "segment_id delta_time latitude"
        # segment ID
        IS2_atl06_dem[gtx]['land_ice_segments']['segment_id'] = val['segment_id'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['segment_id'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['segment_id'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['units'] = "1"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['contentType'] = "referenceInformation"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['long_name'] = "Along-track segment ID number"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['description'] = ("A 7 digit number "
            "identifying the along-track geolocation segment number.  These are sequential, starting with "
            "1 for the first segment after an ascending equatorial crossing node. Equal to the segment_id for "
            "the second of the two 20m ATL03 segments included in the 40m ATL06 segment")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['segment_id']['coordinates'] = \
            "delta_time latitude longitude"

        # dem variables
        IS2_atl06_dem[gtx]['land_ice_segments']['dem'] = {}
        IS2_atl06_fill[gtx]['land_ice_segments']['dem'] = {}
        IS2_atl06_dims[gtx]['land_ice_segments']['dem'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['Description'] = ("The dem group "
            "contains the reference digital elevation model and geoid heights.")
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['data_rate'] = ("Data within this group "
            "are stored at the land_ice_segments segment rate.")

        # interpolate DEM to segment location
        dem_h.data[:] = R1.__call__(np.c_[Y, X])
        dem_h_sigma.data[:] = np.sqrt(R2.__call__(np.c_[Y, X]))
        dem_ice_area = R3.__call__(np.c_[Y, X])
        # update masks and replace fill values
        dem_h.mask[:] = (dem_ice_area <= 0.0) | (np.abs(dem_h.data) >= 1e4)
        dem_h_sigma.mask[:] = (dem_ice_area <= 0.0) | (np.abs(dem_h.data) >= 1e4)
        dem_h.data[dem_h.mask] = dem_h.fill_value
        dem_h_sigma.data[dem_h_sigma.mask] = dem_h_sigma.fill_value

        # save ATL14 DEM elevation for ground track
        IS2_atl06_dem[gtx]['land_ice_segments']['dem']['dem_h'] = dem_h
        IS2_atl06_fill[gtx]['land_ice_segments']['dem']['dem_h'] = dem_h.fill_value
        IS2_atl06_dims[gtx]['land_ice_segments']['dem']['dem_h'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['units'] = "meters"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['contentType'] = "referenceInformation"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['long_name'] = "DEM Height"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['description'] = \
            "Surface height of the digital elevation model (DEM)"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['source'] = 'ATL14'
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h']['coordinates'] = \
            "../segment_id ../delta_time ../latitude ../longitude"

        # save ATL14 DEM elevation uncertainty for ground track
        IS2_atl06_dem[gtx]['land_ice_segments']['dem']['dem_h_sigma'] = dem_h_sigma
        IS2_atl06_fill[gtx]['land_ice_segments']['dem']['dem_h_sigma'] = dem_h_sigma.fill_value
        IS2_atl06_dims[gtx]['land_ice_segments']['dem']['dem_h_sigma'] = ['delta_time']
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma'] = {}
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['units'] = "meters"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['contentType'] = "referenceInformation"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['long_name'] = "DEM Uncertainty"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['description'] = \
            "Uncertainty in the DEM surface height"
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['source'] = 'ATL14'
        IS2_atl06_dem_attrs[gtx]['land_ice_segments']['dem']['dem_h_sigma']['coordinates'] = \
            "../segment_id ../delta_time ../latitude ../longitude"

    # check that there are any valid beams in the dataset
    if bool([k for k in IS2_atl06_dem.keys() if bool(re.match(r'gt\d[lr]',k))]):
        # output HDF5 files with output DEM
        fargs = ('ATL14',YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS,AUX)
        file_format = '{0}_{1}{2}{3}{4}{5}{6}_{7}{8}{9}_{10}_{11}{12}.h5'
        OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(file_format.format(*fargs))
        # print file information
        logging.info(f'\t{str(OUTPUT_FILE)}')
        # write to output HDF5 file
        HDF5_ATL06_dem_write(IS2_atl06_dem, IS2_atl06_dem_attrs,
            FILENAME=OUTPUT_FILE,
            INPUT=[GRANULE, *DEM_MODEL],
            FILL_VALUE=IS2_atl06_fill,
            DIMENSIONS=IS2_atl06_dims,
            CLOBBER=True)
        # change the permissions mode
        OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the interpolated DEM data for ICESat-2 data to HDF5
def HDF5_ATL06_dem_write(IS2_atl06_dem, IS2_atl06_attrs, INPUT=[],
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
    for k,v in IS2_atl06_dem['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl06_dem.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl06_attrs[gtx][att_name]
        # create land_ice_segments group
        fileID[gtx].create_group('land_ice_segments')
        h5[gtx] = dict(land_ice_segments={})
        for att_name in ['Description','data_rate']:
            att_val = IS2_atl06_attrs[gtx]['land_ice_segments'][att_name]
            fileID[gtx]['land_ice_segments'].attrs[att_name] = att_val

        # segment_id, geolocation, time and height variables
        for k in ['delta_time','latitude','longitude','segment_id']:
            # values and attributes
            v = IS2_atl06_dem[gtx]['land_ice_segments'][k]
            attrs = IS2_atl06_attrs[gtx]['land_ice_segments'][k]
            fillvalue = FILL_VALUE[gtx]['land_ice_segments'][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}'.format(gtx,'land_ice_segments',k)
            if fillvalue:
                h5[gtx]['land_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                    compression='gzip')
            else:
                h5[gtx]['land_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[gtx]['land_ice_segments'][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['land_ice_segments'][k]):
                    h5[gtx]['land_ice_segments'][k].dims[i].attach_scale(
                        h5[gtx]['land_ice_segments'][dim])
            else:
                # make dimension
                h5[gtx]['land_ice_segments'][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['land_ice_segments'][k].attrs[att_name] = att_val

        # add to output variables
        for key in ['dem',]:
            fileID[gtx]['land_ice_segments'].create_group(key)
            h5[gtx]['land_ice_segments'][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl06_attrs[gtx]['land_ice_segments'][key][att_name]
                fileID[gtx]['land_ice_segments'][key].attrs[att_name] = att_val
            for k,v in IS2_atl06_dem[gtx]['land_ice_segments'][key].items():
                # attributes
                attrs = IS2_atl06_attrs[gtx]['land_ice_segments'][key][k]
                fillvalue = FILL_VALUE[gtx]['land_ice_segments'][key][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}/{3}'.format(gtx,'land_ice_segments',key,k)
                if fillvalue:
                    h5[gtx]['land_ice_segments'][key][k] = \
                        fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
                else:
                    h5[gtx]['land_ice_segments'][key][k] = \
                        fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, compression='gzip')
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['land_ice_segments'][key][k]):
                    h5[gtx]['land_ice_segments'][key][k].dims[i].attach_scale(
                        h5[gtx]['land_ice_segments'][dim])
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[gtx]['land_ice_segments'][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Land Ice Height'
    fileID.attrs['summary'] = ('Geophysical parameters for land ice segments '
        'needed to interpret and assess the quality of land height estimates.')
    fileID.attrs['description'] = ('Land ice parameters for each beam.  All '
        'parameters are calculated for the same along-track increments for '
        'each beam and repeat.')
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
    # add attributes for input files
    fileID.attrs['lineage'] = [pathlib.Path(i).name for i in INPUT]
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl06_dem[gtx]['land_ice_segments']['longitude']
        lat = IS2_atl06_dem[gtx]['land_ice_segments']['latitude']
        delta_time = IS2_atl06_dem[gtx]['land_ice_segments']['delta_time']
        # setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time.min() if (delta_time.min() < tmn) else tmn
        tmx = delta_time.max() if (delta_time.max() > tmx) else tmx
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
        description="""Interpolate DEMs to ICESat-2 ATL06 land
            ice height locations
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('file',
        type=pathlib.Path,
        help='ICESat-2 ATL06 file to run')
    # directory with output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # full path to ATL14 digital elevation file
    parser.add_argument('--dem-model','-m',
        type=pathlib.Path, nargs='+', required=True,
        help='ICESat-2 ATL14 DEM file to run')
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
