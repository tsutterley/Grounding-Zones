#!/usr/bin/env python
u"""
compute_geoid_ICESat2_ATL12.py
Written by Tyler Sutterley (05/2024)
Computes geoid undulations for correcting ICESat-2 ocean surface height data

COMMAND LINE OPTIONS:
    -O X, --output-directory X: input/output data directory
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
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/ATL12.py: reads ICESat-2 ocean surface height data files
    utilities.py: download and management utilities for syncing files
    geoid_undulation.py: geoidal undulation at a given latitude and longitude
    read_ICGEM_harmonics.py: reads the coefficients for a given gravity model file
    real_potential.py: real potential at a latitude and height for gravity model
    norm_potential.py: normal potential of an ellipsoid at a latitude and height
    norm_gravity.py: normal gravity of an ellipsoid at a latitude and height
    ref_ellipsoid.py: Computes parameters for a reference ellipsoid
    gauss_weights.py: Computes Gaussian weights as a function of degree

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        use time functions from timescale.time
    Updated 07/2023: using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
        additionally output conversion between tide free and mean tide values
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 04/2021: can use a generically named ATL12 file as input
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: H5py deprecation warning change to use make_scale
    Updated 10/2020: using argparse to set command line parameters
    Updated 08/2020: using python3 compatible regular expressions
    Updated 03/2020: use read_ICESat2_ATL12.py from read-ICESat-2 repository
    Forked 11/2019 from compute_geoid_ICESat2_ATL07.py
    Updated 10/2019: changing Y/N flags to True/False
    Written 04/2019
"""
from __future__ import print_function

import re
import logging
import pathlib
import argparse
import datetime
import numpy as np
import grounding_zones as gz

# attempt imports
geoidtk = gz.utilities.import_dependency('geoid_toolkit')
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read ICESat-2 ocean surface height (ATL12) from NSIDC
# and computes geoid undulation at points
def compute_geoid_ICESat2(model_file, INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    LMAX=None,
    LOVE=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # read gravity model Ylms and change tide to tide free
    model_file = pathlib.Path(model_file).expanduser().absolute()
    Ylms = geoidtk.read_ICGEM_harmonics(model_file, LMAX=LMAX, TIDE='tide_free')
    model = Ylms['modelname']
    R = np.float64(Ylms['radius'])
    GM = np.float64(Ylms['earth_gravity_constant'])
    LMAX = np.int64(Ylms['max_degree'])
    # reference to WGS84 ellipsoid
    REFERENCE = 'WGS84'

    # extract parameters from ICESat-2 ATLAS HDF5 ocean surface file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = \
            rx.findall(GRANULE).pop()
    except:
        # output geoid HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_{model}_GEOID{INPUT_FILE.suffix}'
    else:
        # output geoid HDF5 file for ASAS/NSIDC granules
        args = (PRD,model,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format = '{0}_{1}_GEOID_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
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

    # read data from input file
    IS2_atl12_mds,IS2_atl12_attrs,IS2_atl12_beams = \
        is2tk.io.ATL12.read_granule(INPUT_FILE, ATTRIBUTES=True)

    # copy variables for outputting to HDF5 file
    IS2_atl12_geoid = {}
    IS2_atl12_fill = {}
    IS2_atl12_dims = {}
    IS2_atl12_geoid_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl12_geoid['ancillary_data'] = {}
    IS2_atl12_geoid_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl12_geoid['ancillary_data'][key] = IS2_atl12_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl12_geoid_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl12_attrs['ancillary_data'][key].items():
            IS2_atl12_geoid_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl12_beams):
        # output data dictionaries for beam
        IS2_atl12_geoid[gtx] = dict(ssh_segments={})
        IS2_atl12_fill[gtx] = dict(ssh_segments={})
        IS2_atl12_dims[gtx] = dict(ssh_segments={})
        IS2_atl12_geoid_attrs[gtx] = dict(ssh_segments={})

        # extract segment data
        val = IS2_atl12_mds[gtx]['ssh_segments']
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
        IS2_atl12_geoid_attrs[gtx]['Description'] = IS2_atl12_attrs[gtx]['Description']
        IS2_atl12_geoid_attrs[gtx]['atlas_pce'] = IS2_atl12_attrs[gtx]['atlas_pce']
        IS2_atl12_geoid_attrs[gtx]['atlas_beam_type'] = IS2_atl12_attrs[gtx]['atlas_beam_type']
        IS2_atl12_geoid_attrs[gtx]['groundtrack_id'] = IS2_atl12_attrs[gtx]['groundtrack_id']
        IS2_atl12_geoid_attrs[gtx]['atmosphere_profile'] = IS2_atl12_attrs[gtx]['atmosphere_profile']
        IS2_atl12_geoid_attrs[gtx]['atlas_spot_number'] = IS2_atl12_attrs[gtx]['atlas_spot_number']
        IS2_atl12_geoid_attrs[gtx]['sc_orientation'] = IS2_atl12_attrs[gtx]['sc_orientation']
        # group attributes for ssh_segments
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['Description'] = ("Contains "
            "parameters relating to the calculated surface height.")
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['data_rate'] = ("Data within "
            "this group are stored at the variable ocean processing segment rate.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl12_geoid[gtx]['ssh_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl12_fill[gtx]['ssh_segments']['delta_time'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['delta_time'] = None
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delta_time'] = {}
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delta_time']['standard_name'] = "time"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delta_time']['source'] = "telemetry"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delta_time']['calendar'] = "standard"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delta_time']['description'] = ("Number of "
            "GPS seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch "
            "offset is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS "
            "seconds between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP "
            "epoch. By adding the offset contained within atlas_sdp_gps_epoch to delta time "
            "parameters, the time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delta_time']['coordinates'] = \
            "latitude longitude"
        # latitude
        IS2_atl12_geoid[gtx]['ssh_segments']['latitude'] = val['latitude'].copy()
        IS2_atl12_fill[gtx]['ssh_segments']['latitude'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['latitude'] = ['delta_time']
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude'] = {}
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude']['units'] = "degrees_north"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude']['valid_min'] = -90.0
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude']['valid_max'] = 90.0
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['latitude']['coordinates'] = \
            "delta_time longitude"
        # longitude
        IS2_atl12_geoid[gtx]['ssh_segments']['longitude'] = val['longitude'].copy()
        IS2_atl12_fill[gtx]['ssh_segments']['longitude'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['longitude'] = ['delta_time']
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude'] = {}
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude']['units'] = "degrees_east"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude']['valid_min'] = -180.0
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude']['valid_max'] = 180.0
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['longitude']['coordinates'] = \
            "delta_time latitude"
        # Ocean Segment Duration
        IS2_atl12_geoid[gtx]['ssh_segments']['delt_seg'] = val['delt_seg']
        IS2_atl12_fill[gtx]['ssh_segments']['delt_seg'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['delt_seg'] = ['delta_time']
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delt_seg'] = {}
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delt_seg']['units'] = "seconds"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delt_seg']['contentType'] = \
            "referenceInformation"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delt_seg']['long_name'] = \
            "Ocean Segment Duration"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delt_seg']['description'] = \
            "Time duration segment"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['delt_seg']['coordinates'] = \
            "delta_time latitude longitude"

        # stats variables
        IS2_atl12_geoid[gtx]['ssh_segments']['stats'] = {}
        IS2_atl12_fill[gtx]['ssh_segments']['stats'] = {}
        IS2_atl12_dims[gtx]['ssh_segments']['stats'] = {}
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats'] = {}
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['Description'] = ("Contains parameters "
            "related to quality and corrections on the sea surface height parameters.")
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['data_rate'] = ("Data within this group "
            "are stored at the variable ocean processing segment rate.")

        # geoid undulation
        IS2_atl12_geoid[gtx]['ssh_segments']['stats']['geoid_seg'] = N.astype(np.float64)
        IS2_atl12_fill[gtx]['ssh_segments']['stats']['geoid_seg'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['stats']['geoid_seg'] = ['delta_time']
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg'] = {}
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['units'] = "meters"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['contentType'] = "referenceInformation"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['long_name'] = 'Geoidal_Undulation'
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['description'] = ('Geoidal '
            f'undulation above the {REFERENCE} ellipsoid')
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['source'] = Ylms['modelname']
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['earth_gravity_constant'] = GM
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['radius'] = R
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['degree_of_truncation'] = LMAX
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_seg']['coordinates'] = \
            "../delta_time ../latitude ../longitude"
        # geoid conversion
        IS2_atl12_geoid[gtx]['ssh_segments']['stats']['geoid_free2mean_seg'] = free2mean.copy()
        IS2_atl12_fill[gtx]['ssh_segments']['stats']['geoid_free2mean_seg'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['stats']['geoid_free2mean_seg'] = ['delta_time']
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_free2mean_seg'] = {}
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_free2mean_seg']['units'] = "meters"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_free2mean_seg']['contentType'] = "referenceInformation"
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_free2mean_seg']['long_name'] = ('Geoid_'
            'Free-to-Mean_conversion')
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_free2mean_seg']['description'] = ('Additive '
            'value to convert geoid heights from the tide-free system to the mean-tide system')
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_free2mean_seg']['earth_gravity_constant'] = GM
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_free2mean_seg']['radius'] = R
        IS2_atl12_geoid_attrs[gtx]['ssh_segments']['stats']['geoid_free2mean_seg']['coordinates'] = \
            "../delta_time ../latitude ../longitude"

    # print file information
    logging.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_ATL12_geoid_write(IS2_atl12_geoid, IS2_atl12_geoid_attrs,
        CLOBBER=True, INPUT=GRANULE,
        FILL_VALUE=IS2_atl12_fill, DIMENSIONS=IS2_atl12_dims,
        FILENAME=OUTPUT_FILE)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the geoid values for ICESat-2 data to HDF5
def HDF5_ATL12_geoid_write(IS2_atl12_geoid, IS2_atl12_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=False):
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
    for k,v in IS2_atl12_geoid['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl12_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl12_geoid.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl12_attrs[gtx][att_name]
        # create ssh_segments group
        fileID[gtx].create_group('ssh_segments')
        h5[gtx] = dict(ssh_segments={})
        for att_name in ['Description','data_rate']:
            att_val = IS2_atl12_attrs[gtx]['ssh_segments'][att_name]
            fileID[gtx]['ssh_segments'].attrs[att_name] = att_val

        # delta_time, geolocation and segment description variables
        for k in ['delta_time','latitude','longitude','delt_seg']:
            # values and attributes
            v = IS2_atl12_geoid[gtx]['ssh_segments'][k]
            attrs = IS2_atl12_attrs[gtx]['ssh_segments'][k]
            fillvalue = FILL_VALUE[gtx]['ssh_segments'][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}'.format(gtx,'ssh_segments',k)
            if fillvalue:
                h5[gtx]['ssh_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                    compression='gzip')
            else:
                h5[gtx]['ssh_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[gtx]['ssh_segments'][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['ssh_segments'][k]):
                    h5[gtx]['ssh_segments'][k].dims[i].attach_scale(
                        h5[gtx]['ssh_segments'][dim])
            else:
                # make dimension
                h5[gtx]['ssh_segments'][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['ssh_segments'][k].attrs[att_name] = att_val

        # add to stats variables
        key = 'stats'
        fileID[gtx]['ssh_segments'].create_group(key)
        h5[gtx]['ssh_segments'][key] = {}
        for att_name in ['Description','data_rate']:
            att_val=IS2_atl12_attrs[gtx]['ssh_segments'][key][att_name]
            fileID[gtx]['ssh_segments'][key].attrs[att_name] = att_val
        for k,v in IS2_atl12_geoid[gtx]['ssh_segments'][key].items():
            # attributes
            attrs = IS2_atl12_attrs[gtx]['ssh_segments'][key][k]
            fillvalue = FILL_VALUE[gtx]['ssh_segments'][key][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}/{3}'.format(gtx,'ssh_segments',key,k)
            if fillvalue:
                h5[gtx]['ssh_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[gtx]['ssh_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            # attach dimensions
            for i,dim in enumerate(DIMENSIONS[gtx]['ssh_segments'][key][k]):
                h5[gtx]['ssh_segments'][key][k].dims[i].attach_scale(
                    h5[gtx]['ssh_segments'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['ssh_segments'][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L3A Ocean Surface Height'
    fileID.attrs['summary'] = ('Estimates of the ocean surface tidal parameters '
        'needed to interpret and assess the quality of ocean height estimates.')
    fileID.attrs['description'] = ('Sea Surface Height (SSH) of the global '
        'open ocean including the ice-free seasonal ice zone (SIZ) and '
        'near-coast regions.')
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
    # add attributes for input ATL12 file
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl12_geoid[gtx]['ssh_segments']['longitude']
        lat = IS2_atl12_geoid[gtx]['ssh_segments']['latitude']
        delta_time = IS2_atl12_geoid[gtx]['ssh_segments']['delta_time']
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
        description="""Calculates tidal elevations for correcting ICESat-2 ATL12
            ocean surface height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    # input ICESat-2 ocean surface height files
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL12 file to run')
    # directory with output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # set gravity model file to use
    parser.add_argument('--gravity','-G',
        type=pathlib.Path,
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

    # run for each input ATL12 file
    for FILE in args.infile:
        compute_geoid_ICESat2(args.gravity, FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            LMAX=args.lmax,
            LOVE=args.love,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
