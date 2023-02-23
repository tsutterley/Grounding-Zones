#!/usr/bin/env python
u"""
compute_geoid_ICESat2_ATL07.py
Written by Tyler Sutterley (12/2022)
Computes geoid undulations for correcting ICESat-2 sea ice height data

COMMAND LINE OPTIONS:
    -G X, --gravity X: Gravity model file to use (.gfc format)
    -l X, --lmax X: maximum spherical harmonic degree (level of truncation)
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
    time.py: Utilities for calculating time operations
    convert_delta_time.py: converts from delta time into Julian and year-decimal
    io/ATL07.py: reads ICESat-2 sea ice height data files
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
        refactored ICESat-2 data product read programs under io
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
        additionally output conversion between tide free and mean tide values
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 04/2021: can use a generically named ATL07 file as input
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: H5py deprecation warning change to use make_scale
    Updated 10/2020: using argparse to set command line parameters
    Updated 08/2020: using python3 compatible regular expressions
    Updated 03/2020: use read_ICESat2_ATL07.py from read-ICESat-2 repository
    Forked 11/2019 from compute_geoid_ICESat2_ATL06.py
    Updated 10/2019: changing Y/N flags to True/False
    Written 04/2019
"""
from __future__ import print_function

import sys
import os
import re
import logging
import argparse
import datetime
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
try:
    import icesat2_toolkit as is2tk
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("icesat2_toolkit not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read ICESat-2 sea ice height (ATL07) from NSIDC
# and computes geoid undulation at points
def compute_geoid_ICESat2(model_file, INPUT_FILE, LMAX=None, LOVE=None,
    VERBOSE=False, MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # read data from input file
    logging.info(f'{INPUT_FILE} -->')
    IS2_atl07_mds,IS2_atl07_attrs,IS2_atl07_beams = \
        is2tk.io.ATL07.read_granule(INPUT_FILE, ATTRIBUTES=True)
    DIRECTORY = os.path.dirname(INPUT_FILE)

    # read gravity model Ylms and change tide to tide free
    Ylms = geoidtk.read_ICGEM_harmonics(model_file, LMAX=LMAX, TIDE='tide_free')
    R = np.float64(Ylms['radius'])
    GM = np.float64(Ylms['earth_gravity_constant'])
    LMAX = np.int64(Ylms['max_degree'])
    # reference to WGS84 ellipsoid
    REFERENCE = 'WGS84'

    # extract parameters from ICESat-2 ATLAS HDF5 sea ice file name
    rx = re.compile(r'(processed_)?(ATL\d{2})-(\d{2})_(\d{4})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX=rx.findall(INPUT_FILE).pop()
    except:
        # output geoid HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        args = (fileBasename,Ylms['modelname'],fileExtension)
        OUTPUT_FILE = '{0}_{1}_GEOID{2}'.format(*args)
    else:
        # output geoid HDF5 file for ASAS/NSIDC granules
        args = (PRD,HEM,Ylms['modelname'],YY,MM,DD,HH,MN,SS,TRK,CYCL,SN,RL,VERS,AUX)
        ff = '{0}-{1}_{2}_GEOID_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.h5'
        OUTPUT_FILE = ff.format(*args)

    # copy variables for outputting to HDF5 file
    IS2_atl07_geoid = {}
    IS2_atl07_fill = {}
    IS2_atl07_dims = {}
    IS2_atl07_geoid_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl07_geoid['ancillary_data'] = {}
    IS2_atl07_geoid_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl07_geoid['ancillary_data'][key] = IS2_atl07_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl07_geoid_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl07_attrs['ancillary_data'][key].items():
            IS2_atl07_geoid_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl07_beams):
        # output data dictionaries for beam
        IS2_atl07_geoid[gtx] = dict(sea_ice_segments={})
        IS2_atl07_fill[gtx] = dict(sea_ice_segments={})
        IS2_atl07_dims[gtx] = dict(sea_ice_segments={})
        IS2_atl07_geoid_attrs[gtx] = dict(sea_ice_segments={})

        # extract segment data
        val = IS2_atl07_mds[gtx]['sea_ice_segments']
        # colatitude in radians
        theta = (90.0 - val['latitude'])*np.pi/180.0
        # calculate geoid at coordinates
        N = geoidtk.geoid_undulation(val['latitude'], val['longitude'], REFERENCE,
            Ylms['clm'], Ylms['slm'], LMAX, R, GM, GAUSS=0)
        # calculate offset for converting from tide_free to mean_tide
        # legendre polynomial of degree 2 (unnormalized)
        P2 = 0.5*(3.0*np.cos(theta)**2 - 1.0)
        # from Rapp 1991 (Consideration of Permanent Tidal Deformation)
        free2mean = -0.198*P2*(1.0 + LOVE)

        # group attributes for beam
        IS2_atl07_geoid_attrs[gtx]['Description'] = IS2_atl07_attrs[gtx]['Description']
        IS2_atl07_geoid_attrs[gtx]['atlas_pce'] = IS2_atl07_attrs[gtx]['atlas_pce']
        IS2_atl07_geoid_attrs[gtx]['atlas_beam_type'] = IS2_atl07_attrs[gtx]['atlas_beam_type']
        IS2_atl07_geoid_attrs[gtx]['groundtrack_id'] = IS2_atl07_attrs[gtx]['groundtrack_id']
        IS2_atl07_geoid_attrs[gtx]['atmosphere_profile'] = IS2_atl07_attrs[gtx]['atmosphere_profile']
        IS2_atl07_geoid_attrs[gtx]['atlas_spot_number'] = IS2_atl07_attrs[gtx]['atlas_spot_number']
        IS2_atl07_geoid_attrs[gtx]['sc_orientation'] = IS2_atl07_attrs[gtx]['sc_orientation']
        # group attributes for sea_ice_segments
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['Description'] = ("Top group for sea "
            "ice segments as computed by the ATBD algorithm.")
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['data_rate'] = ("Data within this "
            "group are stored at the variable segment rate.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl07_geoid[gtx]['sea_ice_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['delta_time'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['delta_time'] = None
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['delta_time'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['delta_time']['standard_name'] = "time"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['delta_time']['source'] = "telemetry"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['delta_time']['calendar'] = "standard"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['delta_time']['description'] = ("Number of "
            "GPS seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch "
            "offset is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS "
            "seconds between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP "
            "epoch. By adding the offset contained within atlas_sdp_gps_epoch to delta time "
            "parameters, the time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['delta_time']['coordinates'] = \
            "height_segment_id latitude longitude"
        # latitude
        IS2_atl07_geoid[gtx]['sea_ice_segments']['latitude'] = val['latitude'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['latitude'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['latitude'] = ['delta_time']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude']['units'] = "degrees_north"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude']['valid_min'] = -90.0
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude']['valid_max'] = 90.0
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['latitude']['coordinates'] = \
            "height_segment_id delta_time longitude"
        # longitude
        IS2_atl07_geoid[gtx]['sea_ice_segments']['longitude'] = val['longitude'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['longitude'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['longitude'] = ['delta_time']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude']['units'] = "degrees_east"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude']['valid_min'] = -180.0
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude']['valid_max'] = 180.0
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['longitude']['coordinates'] = \
            "height_segment_id delta_time latitude"
        # segment ID
        IS2_atl07_geoid[gtx]['sea_ice_segments']['height_segment_id'] = val['height_segment_id']
        IS2_atl07_fill[gtx]['sea_ice_segments']['height_segment_id'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['height_segment_id'] = ['delta_time']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['height_segment_id'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['height_segment_id']['units'] = "1"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['height_segment_id']['contentType'] = "referenceInformation"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['height_segment_id']['long_name'] = \
            "Identifier of each height segment"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['height_segment_id']['description'] = \
            "Identifier of each height segment"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['height_segment_id']['coordinates'] = \
            "delta_time latitude longitude"
        # geolocation segment beginning
        IS2_atl07_geoid[gtx]['sea_ice_segments']['geoseg_beg'] = val['geoseg_beg'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geoseg_beg'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geoseg_beg'] = ['delta_time']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_beg'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_beg']['units'] = "1"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_beg']['contentType'] = "referenceInformation"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_beg']['long_name'] = "Beginning GEOSEG"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_beg']['description'] = \
            "Geolocation segment (geoseg) ID associated with the first photon used in this sea ice segment"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_beg']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"
        # geolocation segment ending
        IS2_atl07_geoid[gtx]['sea_ice_segments']['geoseg_end'] = val['geoseg_end'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geoseg_end'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geoseg_end'] = ['delta_time']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_end'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_end']['units'] = "1"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_end']['contentType'] = "referenceInformation"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_end']['long_name'] = "Ending GEOSEG"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_end']['description'] = \
            "Geolocation segment (geoseg) ID associated with the last photon used in this sea ice segment"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geoseg_end']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"
        # along track distance
        IS2_atl07_geoid[gtx]['sea_ice_segments']['seg_dist_x'] = val['seg_dist_x'].copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['seg_dist_x'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['seg_dist_x'] = ['delta_time']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['seg_dist_x'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['seg_dist_x']['units'] = "meters"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['seg_dist_x']['contentType'] = "referenceInformation"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['seg_dist_x']['long_name'] = "Along track distance"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['seg_dist_x']['description'] = \
            "Along-track distance from the equator crossing to the segment center."
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['seg_dist_x']['coordinates'] = \
            "height_segment_id delta_time latitude longitude"

        # geophysical variables
        IS2_atl07_geoid[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['Description'] = ("Contains geophysical "
            "parameters and corrections used to correct photon heights for geophysical effects, such as tides.")
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['data_rate'] = ("Data within this group "
            "are stored at the sea_ice_height segment rate.")

        # geoid undulation
        IS2_atl07_geoid[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid'] = N.astype(np.float64)
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid'] = ['delta_time']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['units'] = "meters"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['contentType'] = "referenceInformation"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['long_name'] = 'Geoidal_Undulation'
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['description'] = ('Geoidal '
            f'undulation above the {REFERENCE} ellipsoid')
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['source'] = Ylms['modelname']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['earth_gravity_constant'] = GM
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['radius'] = R
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['degree_of_truncation'] = LMAX
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid']['coordinates'] = \
            "../height_segment_id ../delta_time ../latitude ../longitude"
        # geoid conversion
        IS2_atl07_geoid[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean'] = free2mean.copy()
        IS2_atl07_fill[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean'] = None
        IS2_atl07_dims[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean'] = ['delta_time']
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean'] = {}
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean']['units'] = "meters"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean']['contentType'] = "referenceInformation"
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean']['long_name'] = ('Geoid_'
            'Free-to-Mean_conversion')
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean']['description'] = ('Additive '
            'value to convert geoid heights from the tide-free system to the mean-tide system')
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean']['earth_gravity_constant'] = GM
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean']['radius'] = R
        IS2_atl07_geoid_attrs[gtx]['sea_ice_segments']['geophysical']['height_segment_geoid_free2mean']['coordinates'] = \
            "../height_segment_id ../delta_time ../latitude ../longitude"

    # print file information
    logging.info(f'\t{OUTPUT_FILE}')
    HDF5_ATL07_geoid_write(IS2_atl07_geoid, IS2_atl07_geoid_attrs,
        CLOBBER=True, INPUT=os.path.basename(INPUT_FILE),
        FILL_VALUE=IS2_atl07_fill, DIMENSIONS=IS2_atl07_dims,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE))
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

# PURPOSE: outputting the geoid values for ICESat-2 data to HDF5
def HDF5_ATL07_geoid_write(IS2_atl07_geoid, IS2_atl07_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    # create HDF5 records
    h5 = {}

    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl07_geoid['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl07_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl07_geoid.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl07_attrs[gtx][att_name]
        # create sea_ice_segments group
        fileID[gtx].create_group('sea_ice_segments')
        h5[gtx] = dict(sea_ice_segments={})
        for att_name in ['Description','data_rate']:
            att_val = IS2_atl07_attrs[gtx]['sea_ice_segments'][att_name]
            fileID[gtx]['sea_ice_segments'].attrs[att_name] = att_val

        # delta_time, geolocation and segment identification variables
        for k in ['delta_time','latitude','longitude','height_segment_id',
            'geoseg_beg','geoseg_end','seg_dist_x']:
            # values and attributes
            v = IS2_atl07_geoid[gtx]['sea_ice_segments'][k]
            attrs = IS2_atl07_attrs[gtx]['sea_ice_segments'][k]
            fillvalue = FILL_VALUE[gtx]['sea_ice_segments'][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}'.format(gtx,'sea_ice_segments',k)
            if fillvalue:
                h5[gtx]['sea_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                    compression='gzip')
            else:
                h5[gtx]['sea_ice_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[gtx]['sea_ice_segments'][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['sea_ice_segments'][k]):
                    h5[gtx]['sea_ice_segments'][k].dims[i].attach_scale(
                        h5[gtx]['sea_ice_segments'][dim])
            else:
                # make dimension
                h5[gtx]['sea_ice_segments'][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['sea_ice_segments'][k].attrs[att_name] = att_val

        # add to geophysical corrections
        key = 'geophysical'
        fileID[gtx]['sea_ice_segments'].create_group(key)
        h5[gtx]['sea_ice_segments'][key] = {}
        for att_name in ['Description','data_rate']:
            att_val=IS2_atl07_attrs[gtx]['sea_ice_segments'][key][att_name]
            fileID[gtx]['sea_ice_segments'][key].attrs[att_name] = att_val
        for k,v in IS2_atl07_geoid[gtx]['sea_ice_segments'][key].items():
            # attributes
            attrs = IS2_atl07_attrs[gtx]['sea_ice_segments'][key][k]
            fillvalue = FILL_VALUE[gtx]['sea_ice_segments'][key][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}/{3}'.format(gtx,'sea_ice_segments',key,k)
            if fillvalue:
                h5[gtx]['sea_ice_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[gtx]['sea_ice_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            # attach dimensions
            for i,dim in enumerate(DIMENSIONS[gtx]['sea_ice_segments'][key][k]):
                h5[gtx]['sea_ice_segments'][key][k].dims[i].attach_scale(
                    h5[gtx]['sea_ice_segments'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['sea_ice_segments'][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L3A Sea Ice Height'
    fileID.attrs['summary'] = ('Estimates of the sea ice tidal parameters '
        'needed to interpret and assess the quality of sea height estimates.')
    fileID.attrs['description'] = ('The data set (ATL07) contains along-track '
        'heights for sea ice and open water leads (at varying length scales) '
        'relative to the WGS84 ellipsoid (ITRF2014 reference frame) after '
        'adjustment for geoidal and tidal variations, and inverted barometer '
        'effects.')
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
    # add attributes for input ATL07 file
    fileID.attrs['input_files'] = os.path.basename(INPUT)
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl07_geoid[gtx]['sea_ice_segments']['longitude']
        lat = IS2_atl07_geoid[gtx]['sea_ice_segments']['latitude']
        delta_time = IS2_atl07_geoid[gtx]['sea_ice_segments']['delta_time']
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
    # convert start and end time from ATLAS SDP seconds into UTC time
    time_utc = is2tk.convert_delta_time(np.array([tmn,tmx]))
    # convert to calendar date
    YY,MM,DD,HH,MN,SS = is2tk.time.convert_julian(time_utc['julian'],
        FORMAT='tuple')
    # add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = f'{tmx-tmn:0.0f}'
    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates tidal elevations for correcting ICESat-2 ATL07
            sea ice height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    # input ICESat-2 sea ice height files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL07 file to run')
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

    # run for each input ATL07 file
    for FILE in args.infile:
        compute_geoid_ICESat2(args.gravity, FILE, LMAX=args.lmax,
            LOVE=args.love, VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
