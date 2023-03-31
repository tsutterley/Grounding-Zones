#!/usr/bin/env python
u"""
compute_geoid_ICESat2_ATL11.py
Written by Tyler Sutterley (12/2022)
Computes geoid undulations for correcting ICESat-2 annual land ice height data

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
    io/ATL11.py: reads ICESat-2 annual land ice height data files
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
    Updated 04/2021: can use a generically named ATL11 file as input
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Written 12/2020

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
import collections
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

# PURPOSE: read ICESat-2 annual land ice height data (ATL11) from NSIDC
# and computes geoid undulation at points
def compute_geoid_ICESat2(model_file, INPUT_FILE, LMAX=None, LOVE=None,
    VERBOSE=False, MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # read data from input file
    logging.info(f'{INPUT_FILE} -->')
    IS2_atl11_mds,IS2_atl11_attrs,IS2_atl11_pairs = \
        is2tk.io.ATL11.read_granule(INPUT_FILE,
                                    ATTRIBUTES=True,
                                    CROSSOVERS=True)
    DIRECTORY = os.path.dirname(INPUT_FILE)

    # read gravity model Ylms and change tide to tide free
    Ylms = geoidtk.read_ICGEM_harmonics(model_file, LMAX=LMAX, TIDE='tide_free')
    R = np.float64(Ylms['radius'])
    GM = np.float64(Ylms['earth_gravity_constant'])
    LMAX = np.int64(Ylms['max_degree'])
    # reference to WGS84 ellipsoid
    REFERENCE = 'WGS84'

    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(INPUT_FILE).pop()
    except:
        # output geoid HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        args = (fileBasename,Ylms['modelname'],fileExtension)
        OUTPUT_FILE = '{0}_{1}_GEOID{2}'.format(*args)
    else:
        # output geoid HDF5 file for ASAS/NSIDC granules
        args = (PRD,Ylms['modelname'],TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        file_format = '{0}_{1}_GEOID_{2}{3}_{4}{5}_{6}_{7}{8}.h5'
        OUTPUT_FILE = file_format.format(*args)

    # copy variables for outputting to HDF5 file
    IS2_atl11_geoid = {}
    IS2_atl11_fill = {}
    IS2_atl11_dims = {}
    IS2_atl11_geoid_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl11_geoid['ancillary_data'] = {}
    IS2_atl11_geoid_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl11_geoid['ancillary_data'][key] = IS2_atl11_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl11_geoid_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][key].items():
            IS2_atl11_geoid_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam pair within the file
    for ptx in sorted(IS2_atl11_pairs):
        # output data dictionaries for beam
        IS2_atl11_geoid[ptx] = dict(cycle_stats=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(cycle_stats={})
        IS2_atl11_dims[ptx] = dict(cycle_stats={})
        IS2_atl11_geoid_attrs[ptx] = dict(cycle_stats={})

        # along-track (AT) reference point, latitude, longitude and time
        ref_pt = IS2_atl11_mds[ptx]['ref_pt'].copy()
        latitude = np.ma.array(IS2_atl11_mds[ptx]['latitude'],
            fill_value=IS2_atl11_attrs[ptx]['latitude']['_FillValue'])
        longitude = np.ma.array(IS2_atl11_mds[ptx]['longitude'],
            fill_value=IS2_atl11_attrs[ptx]['longitude']['_FillValue'])
        delta_time = np.ma.array(IS2_atl11_mds[ptx]['delta_time'],
            fill_value=IS2_atl11_attrs[ptx]['delta_time']['_FillValue'])

        # number of average segments and number of included cycles
        # fill_value for invalid heights and corrections
        fv = IS2_atl11_attrs[ptx]['h_corr']['_FillValue']
        # shape of along-track and across-track data
        n_points,n_cycles = delta_time.shape

        # colatitude in radians
        theta = (90.0 - latitude)*np.pi/180.0
        # allocate for output geoid undulation
        N = np.ma.zeros((n_points),fill_value=fv)
        N.mask = (latitude.data == latitude.fill_value)
        valid, = np.nonzero(np.logical_not(N.mask))
        # calculate geoid at coordinates
        N.data[valid] = geoidtk.geoid_undulation(latitude.data[valid],
            longitude.data[valid], REFERENCE,
            Ylms['clm'], Ylms['slm'],
            LMAX, R, GM).astype(np.float64)
        # calculate offset for converting from tide_free to mean_tide
        # legendre polynomial of degree 2 (unnormalized)
        P2 = 0.5*(3.0*np.cos(theta)**2 - 1.0)
        # from Rapp 1991 (Consideration of Permanent Tidal Deformation)
        free2mean = np.ma.zeros((n_points),fill_value=fv)
        free2mean.data[valid] = -0.198*P2[valid]*(1.0 + LOVE)
        free2mean.mask = np.copy(N.mask)
        # replace invalid values with fill value
        N.data[N.mask] = N.fill_value
        free2mean.data[free2mean.mask] = free2mean.fill_value

        # group attributes for beam
        IS2_atl11_geoid_attrs[ptx]['description'] = ('Contains the primary science parameters '
            'for this data set')
        IS2_atl11_geoid_attrs[ptx]['beam_pair'] = IS2_atl11_attrs[ptx]['beam_pair']
        IS2_atl11_geoid_attrs[ptx]['ReferenceGroundTrack'] = IS2_atl11_attrs[ptx]['ReferenceGroundTrack']
        IS2_atl11_geoid_attrs[ptx]['first_cycle'] = IS2_atl11_attrs[ptx]['first_cycle']
        IS2_atl11_geoid_attrs[ptx]['last_cycle'] = IS2_atl11_attrs[ptx]['last_cycle']
        IS2_atl11_geoid_attrs[ptx]['equatorial_radius'] = IS2_atl11_attrs[ptx]['equatorial_radius']
        IS2_atl11_geoid_attrs[ptx]['polar_radius'] = IS2_atl11_attrs[ptx]['polar_radius']

        # geolocation, time and reference point
        # reference point
        IS2_atl11_geoid[ptx]['ref_pt'] = ref_pt.copy()
        IS2_atl11_fill[ptx]['ref_pt'] = None
        IS2_atl11_dims[ptx]['ref_pt'] = None
        IS2_atl11_geoid_attrs[ptx]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_geoid_attrs[ptx]['ref_pt']['units'] = "1"
        IS2_atl11_geoid_attrs[ptx]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_geoid_attrs[ptx]['ref_pt']['long_name'] = "Reference point number"
        IS2_atl11_geoid_attrs[ptx]['ref_pt']['source'] = "ATL06"
        IS2_atl11_geoid_attrs[ptx]['ref_pt']['description'] = ("The reference point is the "
            "7 digit segment_id number corresponding to the center of the ATL06 data used "
            "for each ATL11 point.  These are sequential, starting with 1 for the first "
            "segment after an ascending equatorial crossing node.")
        IS2_atl11_geoid_attrs[ptx]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number
        IS2_atl11_geoid[ptx]['cycle_number'] = IS2_atl11_mds[ptx]['cycle_number'].copy()
        IS2_atl11_fill[ptx]['cycle_number'] = None
        IS2_atl11_dims[ptx]['cycle_number'] = None
        IS2_atl11_geoid_attrs[ptx]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_geoid_attrs[ptx]['cycle_number']['units'] = "1"
        IS2_atl11_geoid_attrs[ptx]['cycle_number']['long_name'] = "Orbital cycle number"
        IS2_atl11_geoid_attrs[ptx]['cycle_number']['source'] = "ATL06"
        IS2_atl11_geoid_attrs[ptx]['cycle_number']['description'] = ("Number of 91-day periods "
            "that have elapsed since ICESat-2 entered the science orbit. Each of the 1,387 "
            "reference ground track (RGTs) is targeted in the polar regions once "
            "every 91 days.")
        # delta time
        IS2_atl11_geoid[ptx]['delta_time'] = delta_time.copy()
        IS2_atl11_fill[ptx]['delta_time'] = delta_time.fill_value
        IS2_atl11_dims[ptx]['delta_time'] = ['ref_pt','cycle_number']
        IS2_atl11_geoid_attrs[ptx]['delta_time'] = collections.OrderedDict()
        IS2_atl11_geoid_attrs[ptx]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_geoid_attrs[ptx]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_geoid_attrs[ptx]['delta_time']['standard_name'] = "time"
        IS2_atl11_geoid_attrs[ptx]['delta_time']['calendar'] = "standard"
        IS2_atl11_geoid_attrs[ptx]['delta_time']['source'] = "ATL06"
        IS2_atl11_geoid_attrs[ptx]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_geoid_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt cycle_number latitude longitude"
        # latitude
        IS2_atl11_geoid[ptx]['latitude'] = latitude.copy()
        IS2_atl11_fill[ptx]['latitude'] = latitude.fill_value
        IS2_atl11_dims[ptx]['latitude'] = ['ref_pt']
        IS2_atl11_geoid_attrs[ptx]['latitude'] = collections.OrderedDict()
        IS2_atl11_geoid_attrs[ptx]['latitude']['units'] = "degrees_north"
        IS2_atl11_geoid_attrs[ptx]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_geoid_attrs[ptx]['latitude']['long_name'] = "Latitude"
        IS2_atl11_geoid_attrs[ptx]['latitude']['standard_name'] = "latitude"
        IS2_atl11_geoid_attrs[ptx]['latitude']['source'] = "ATL06"
        IS2_atl11_geoid_attrs[ptx]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_geoid_attrs[ptx]['latitude']['valid_min'] = -90.0
        IS2_atl11_geoid_attrs[ptx]['latitude']['valid_max'] = 90.0
        IS2_atl11_geoid_attrs[ptx]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude
        IS2_atl11_geoid[ptx]['longitude'] = longitude.copy()
        IS2_atl11_fill[ptx]['longitude'] = longitude.fill_value
        IS2_atl11_dims[ptx]['longitude'] = ['ref_pt']
        IS2_atl11_geoid_attrs[ptx]['longitude'] = collections.OrderedDict()
        IS2_atl11_geoid_attrs[ptx]['longitude']['units'] = "degrees_east"
        IS2_atl11_geoid_attrs[ptx]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_geoid_attrs[ptx]['longitude']['long_name'] = "Longitude"
        IS2_atl11_geoid_attrs[ptx]['longitude']['standard_name'] = "longitude"
        IS2_atl11_geoid_attrs[ptx]['longitude']['source'] = "ATL06"
        IS2_atl11_geoid_attrs[ptx]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_geoid_attrs[ptx]['longitude']['valid_min'] = -180.0
        IS2_atl11_geoid_attrs[ptx]['longitude']['valid_max'] = 180.0
        IS2_atl11_geoid_attrs[ptx]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"

        # reference surface variables
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['Description'] = ("The ref_surf subgroup contains "
            "parameters that describe the reference surface fit at each reference point, "
            "including slope information from ATL06, the polynomial coefficients used for the "
            "fit, and misfit statistics.")
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['data_rate'] = ("Data within this group "
            "are stored at the average segment rate.")
        # geoid undulation
        IS2_atl11_geoid[ptx]['ref_surf']['geoid_h'] = N.astype(np.float64)
        IS2_atl11_fill[ptx]['ref_surf']['geoid_h'] = N.fill_value
        IS2_atl11_dims[ptx]['ref_surf']['geoid_h'] = ['ref_pt']
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h'] = collections.OrderedDict()
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['units'] = "meters"
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['contentType'] = "referenceInformation"
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['long_name'] = 'Geoidal_Undulation'
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['description'] = ('Geoidal '
            f'undulation above the {REFERENCE} ellipsoid')
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['source'] = Ylms['modelname']
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['earth_gravity_constant'] = GM
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['radius'] = R
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['degree_of_truncation'] = LMAX
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_h']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"
        # geoid conversion
        IS2_atl11_geoid[ptx]['ref_surf']['geoid_free2mean'] = free2mean.copy()
        IS2_atl11_fill[ptx]['ref_surf']['geoid_free2mean'] = free2mean.fill_value
        IS2_atl11_dims[ptx]['ref_surf']['geoid_free2mean'] = ['ref_pt']
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_free2mean'] = collections.OrderedDict()
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_free2mean']['units'] = "meters"
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_free2mean']['contentType'] = "referenceInformation"
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_free2mean']['long_name'] = ('Geoid_'
            'Free-to-Mean_conversion')
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_free2mean']['description'] = ('Additive '
            'value to convert geoid heights from the tide-free system to the mean-tide system')
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_free2mean']['earth_gravity_constant'] = GM
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_free2mean']['radius'] = R
        IS2_atl11_geoid_attrs[ptx]['ref_surf']['geoid_free2mean']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"

    # print file information
    logging.info(f'\t{os.path.join(DIRECTORY,OUTPUT_FILE)}')
    HDF5_ATL11_geoid_write(IS2_atl11_geoid, IS2_atl11_geoid_attrs,
        CLOBBER=True, INPUT=os.path.basename(INPUT_FILE),
        FILL_VALUE=IS2_atl11_fill, DIMENSIONS=IS2_atl11_dims,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE))
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

# PURPOSE: outputting the geoid values for ICESat-2 data to HDF5
def HDF5_ATL11_geoid_write(IS2_atl11_geoid, IS2_atl11_attrs, INPUT=None,
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
    for k,v in IS2_atl11_geoid['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam pair
    pairs = [k for k in IS2_atl11_geoid.keys() if bool(re.match(r'pt\d',k))]
    for ptx in pairs:
        fileID.create_group(ptx)
        h5[ptx] = {}
        # add HDF5 group attributes for beam
        for att_name in ['description','beam_pair','ReferenceGroundTrack',
            'first_cycle','last_cycle','equatorial_radius','polar_radius']:
            fileID[ptx].attrs[att_name] = IS2_atl11_attrs[ptx][att_name]

        # ref_pt, cycle number, geolocation and delta_time variables
        for k in ['ref_pt','cycle_number','delta_time','latitude','longitude']:
            # values and attributes
            v = IS2_atl11_geoid[ptx][k]
            attrs = IS2_atl11_attrs[ptx][k]
            fillvalue = FILL_VALUE[ptx][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}'.format(ptx,k)
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

        # add to ref_surf variables
        for key in ['ref_surf']:
            fileID[ptx].create_group(key)
            h5[ptx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl11_attrs[ptx][key][att_name]
                fileID[ptx][key].attrs[att_name] = att_val
            for k,v in IS2_atl11_geoid[ptx][key].items():
                # attributes
                attrs = IS2_atl11_attrs[ptx][key][k]
                fillvalue = FILL_VALUE[ptx][key][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(ptx,key,k)
                if fillvalue:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
                else:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, compression='gzip')
                # create or attach dimensions for HDF5 variable
                if DIMENSIONS[ptx][key][k]:
                    # attach dimensions
                    for i,dim in enumerate(DIMENSIONS[ptx][key][k]):
                        h5[ptx][key][k].dims[i].attach_scale(h5[ptx][dim])
                else:
                    # make dimension
                    h5[ptx][key][k].make_scale(k)
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[ptx][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Annual Land Ice Height'
    fileID.attrs['summary'] = ('The purpose of ATL11 is to provide an ICESat-2 '
        'satellite cycle summary of heights and height changes of land-based '
        'ice and will be provided as input to ATL15 and ATL16, gridded '
        'estimates of heights and height-changes.')
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
    fileID.attrs['input_files'] = os.path.basename(INPUT)
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for ptx in pairs:
        lon = IS2_atl11_geoid[ptx]['longitude']
        lat = IS2_atl11_geoid[ptx]['latitude']
        delta_time = IS2_atl11_geoid[ptx]['delta_time']
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
        description="""Calculates tidal elevations for correcting ICESat-2 ATL11
            annual land ice height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    # input ICESat-2 annual land ice height files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL11 file to run')
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

    # run for each input ATL11 file
    for FILE in args.infile:
        compute_geoid_ICESat2(args.gravity, FILE, LMAX=args.lmax,
            LOVE=args.love, VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
