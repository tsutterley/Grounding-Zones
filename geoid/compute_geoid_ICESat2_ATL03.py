#!/usr/bin/env python
u"""
compute_geoid_ICESat2_ATL03.py
Written by Tyler Sutterley (05/2022)
Computes geoid undulations for correcting ICESat-2 photon height data
Calculated at ATL03 segment level using reference photon geolocation and time
Segment level corrections can be applied to the individual photon events (PEs)

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
    read_ICESat2_ATL03.py: reads ICESat-2 global geolocated photon data files
    utilities.py: download and management utilities for syncing files
    geoid_undulation.py: geoidal undulation at a given latitude and longitude
    read_ICGEM_harmonics.py: reads the coefficients for a given gravity model file
    real_potential.py: real potential at a latitude and height for gravity model
    norm_potential.py: normal potential of an ellipsoid at a latitude and height
    norm_gravity.py: normal gravity of an ellipsoid at a latitude and height
    ref_ellipsoid.py: Computes parameters for a reference ellipsoid
    gauss_weights.py: Computes Gaussian weights as a function of degree

UPDATE HISTORY:
    Updated 05/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
        additionally output conversion between tide free and mean tide values
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 04/2021: can use a generically named ATL03 file as input
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: H5py deprecation warning change to use make_scale
    Updated 10/2020: using argparse to set command line parameters
    Updated 08/2020: using python3 compatible regular expressions
    Updated 03/2020: use read_ICESat2_ATL03.py from read-ICESat-2 repository
    Updated 10/2019: changing Y/N flags to True/False
    Written 04/2019
"""
from __future__ import print_function

import sys
import os
import re
import h5py
import logging
import argparse
import datetime
import numpy as np
import icesat2_toolkit.time
from geoid_toolkit.read_ICGEM_harmonics import read_ICGEM_harmonics
from geoid_toolkit.geoid_undulation import geoid_undulation
from grounding_zones.utilities import convert_arg_line_to_args
from icesat2_toolkit.convert_delta_time import convert_delta_time
from icesat2_toolkit.read_ICESat2_ATL03 import read_HDF5_ATL03_main, \
    read_HDF5_ATL03_beam

#-- PURPOSE: read ICESat-2 geolocated photon data (ATL03) from NSIDC
#-- and computes geoid undulation at points
def compute_geoid_ICESat2(model_file, INPUT_FILE, LMAX=None, LOVE=None,
    VERBOSE=False, MODE=0o775):

    #-- create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- read data from input file
    logging.info('{0} -->'.format(INPUT_FILE))
    IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams = read_HDF5_ATL03_main(INPUT_FILE,
        ATTRIBUTES=True)
    DIRECTORY = os.path.dirname(INPUT_FILE)

    #-- read gravity model Ylms and change tide to tide free
    Ylms = read_ICGEM_harmonics(model_file, LMAX=LMAX, TIDE='tide_free')
    R = np.float64(Ylms['radius'])
    GM = np.float64(Ylms['earth_gravity_constant'])
    LMAX = np.int64(Ylms['max_degree'])
    #-- reference to WGS84 ellipsoid
    REFERENCE = 'WGS84'

    #-- extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = rx.findall(INPUT_FILE).pop()
    except:
        #-- output geoid HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        args = (fileBasename,Ylms['modelname'],fileExtension)
        OUTPUT_FILE = '{0}_{1}_GEOID{2}'.format(*args)
    else:
        #-- output geoid HDF5 file for ASAS/NSIDC granules
        args = (PRD,Ylms['modelname'],YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format = '{0}_{1}_GEOID_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
        OUTPUT_FILE = file_format.format(*args)

    #-- copy variables for outputting to HDF5 file
    IS2_atl03_geoid = {}
    IS2_atl03_fill = {}
    IS2_atl03_dims = {}
    IS2_atl03_geoid_attrs = {}
    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    #-- Add this value to delta time parameters to compute full gps_seconds
    IS2_atl03_geoid['ancillary_data'] = {}
    IS2_atl03_geoid_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        #-- get each HDF5 variable
        IS2_atl03_geoid['ancillary_data'][key] = IS2_atl03_mds['ancillary_data'][key]
        #-- Getting attributes of group and included variables
        IS2_atl03_geoid_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][key].items():
            IS2_atl03_geoid_attrs['ancillary_data'][key][att_name] = att_val

    #-- for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        #-- output data dictionaries for beam
        IS2_atl03_geoid[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_fill[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_dims[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_geoid_attrs[gtx] = dict(geolocation={}, geophys_corr={})

        #-- read data and attributes for beam
        val,attrs = read_HDF5_ATL03_beam(INPUT_FILE,gtx,ATTRIBUTES=True)
        #-- extract variables for computing geoid heights
        segment_id = val['geolocation']['segment_id'].copy()
        delta_time = val['geolocation']['delta_time'].copy()
        lon = val['geolocation']['reference_photon_lon'].copy()
        lat = val['geolocation']['reference_photon_lat'].copy()
        #-- colatitude in radians
        theta = (90.0 - lat)*np.pi/180.0

        #-- calculate geoid at coordinates
        N = geoid_undulation(lat, lon, REFERENCE,
            Ylms['clm'], Ylms['slm'], LMAX, R, GM, GAUSS=0)
        #-- calculate offset for converting from tide_free to mean_tide
        #-- legendre polynomial of degree 2 (unnormalized)
        P2 = 0.5*(3.0*np.cos(theta)**2 - 1.0)
        #-- from Rapp 1991 (Consideration of Permanent Tidal Deformation)
        free2mean = -0.198*P2*(1.0 + LOVE)

        #-- group attributes for beam
        IS2_atl03_geoid_attrs[gtx]['Description'] = attrs['Description']
        IS2_atl03_geoid_attrs[gtx]['atlas_pce'] = attrs['atlas_pce']
        IS2_atl03_geoid_attrs[gtx]['atlas_beam_type'] = attrs['atlas_beam_type']
        IS2_atl03_geoid_attrs[gtx]['groundtrack_id'] = attrs['groundtrack_id']
        IS2_atl03_geoid_attrs[gtx]['atmosphere_profile'] = attrs['atmosphere_profile']
        IS2_atl03_geoid_attrs[gtx]['atlas_spot_number'] = attrs['atlas_spot_number']
        IS2_atl03_geoid_attrs[gtx]['sc_orientation'] = attrs['sc_orientation']

        #-- group attributes for geolocation
        IS2_atl03_geoid_attrs[gtx]['geolocation']['Description'] = ("Contains parameters related to "
            "geolocation.  The rate of all of these parameters is at the rate corresponding to the "
            "ICESat-2 Geolocation Along Track Segment interval (nominally 20 m along-track).")
        IS2_atl03_geoid_attrs[gtx]['geolocation']['data_rate'] = ("Data within this group are "
            "stored at the ICESat-2 20m segment rate.")
        #-- group attributes for geophys_corr
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['Description'] = ("Contains parameters used to "
            "correct photon heights for geophysical effects, such as tides.  These parameters are "
            "posted at the same interval as the ICESat-2 Geolocation Along-Track Segment interval "
            "(nominally 20m along-track).")
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['data_rate'] = ("These parameters are stored at "
            "the ICESat-2 Geolocation Along Track Segment rate (nominally every 20 m along-track).")

        #-- geolocation, time and segment ID
        #-- delta time in geolocation group
        IS2_atl03_geoid[gtx]['geolocation']['delta_time'] = delta_time
        IS2_atl03_fill[gtx]['geolocation']['delta_time'] = None
        IS2_atl03_dims[gtx]['geolocation']['delta_time'] = None
        IS2_atl03_geoid_attrs[gtx]['geolocation']['delta_time'] = {}
        IS2_atl03_geoid_attrs[gtx]['geolocation']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['delta_time']['standard_name'] = "time"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['delta_time']['calendar'] = "standard"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['delta_time']['description'] = ("Elapsed seconds "
            "from the ATLAS SDP GPS Epoch, corresponding to the transmit time of the reference "
            "photon. The ATLAS Standard Data Products (SDP) epoch offset is defined within "
            "/ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds between the GPS epoch "
            "(1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By adding the offset "
            "contained within atlas_sdp_gps_epoch to delta time parameters, the time in gps_seconds "
            "relative to the GPS epoch can be computed.")
        IS2_atl03_geoid_attrs[gtx]['geolocation']['delta_time']['coordinates'] = \
            "segment_id reference_photon_lat reference_photon_lon"
        #-- delta time in geophys_corr group
        IS2_atl03_geoid[gtx]['geophys_corr']['delta_time'] = delta_time
        IS2_atl03_fill[gtx]['geophys_corr']['delta_time'] = None
        IS2_atl03_dims[gtx]['geophys_corr']['delta_time'] = None
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['delta_time'] = {}
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['delta_time']['standard_name'] = "time"
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['delta_time']['calendar'] = "standard"
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['delta_time']['description'] = ("Elapsed seconds "
            "from the ATLAS SDP GPS Epoch, corresponding to the transmit time of the reference "
            "photon. The ATLAS Standard Data Products (SDP) epoch offset is defined within "
            "/ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds between the GPS epoch "
            "(1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By adding the offset "
            "contained within atlas_sdp_gps_epoch to delta time parameters, the time in gps_seconds "
            "relative to the GPS epoch can be computed.")
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['delta_time']['coordinates'] = ("../geolocation/segment_id "
            "../geolocation/reference_photon_lat ../geolocation/reference_photon_lon")

        #-- latitude
        IS2_atl03_geoid[gtx]['geolocation']['reference_photon_lat'] = lat
        IS2_atl03_fill[gtx]['geolocation']['reference_photon_lat'] = None
        IS2_atl03_dims[gtx]['geolocation']['reference_photon_lat'] = ['delta_time']
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat'] = {}
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat']['units'] = "degrees_north"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat']['contentType'] = "physicalMeasurement"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat']['long_name'] = "Latitude"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat']['standard_name'] = "latitude"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat']['description'] = ("Latitude of each "
            "reference photon. Computed from the ECF Cartesian coordinates of the bounce point.")
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat']['valid_min'] = -90.0
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat']['valid_max'] = 90.0
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lat']['coordinates'] = \
            "segment_id delta_time reference_photon_lon"
        #-- longitude
        IS2_atl03_geoid[gtx]['geolocation']['reference_photon_lon'] = lon
        IS2_atl03_fill[gtx]['geolocation']['reference_photon_lon'] = None
        IS2_atl03_dims[gtx]['geolocation']['reference_photon_lon'] = ['delta_time']
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon'] = {}
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon']['units'] = "degrees_east"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon']['contentType'] = "physicalMeasurement"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon']['long_name'] = "Longitude"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon']['standard_name'] = "longitude"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon']['description'] = ("Longitude of each "
            "reference photon. Computed from the ECF Cartesian coordinates of the bounce point.")
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon']['valid_min'] = -180.0
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon']['valid_max'] = 180.0
        IS2_atl03_geoid_attrs[gtx]['geolocation']['reference_photon_lon']['coordinates'] = \
            "segment_id delta_time reference_photon_lat"
        #-- segment ID
        IS2_atl03_geoid[gtx]['geolocation']['segment_id'] = segment_id
        IS2_atl03_fill[gtx]['geolocation']['segment_id'] = None
        IS2_atl03_dims[gtx]['geolocation']['segment_id'] = ['delta_time']
        IS2_atl03_geoid_attrs[gtx]['geolocation']['segment_id'] = {}
        IS2_atl03_geoid_attrs[gtx]['geolocation']['segment_id']['units'] = "1"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['segment_id']['contentType'] = "referenceInformation"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['segment_id']['long_name'] = "Along-track segment ID number"
        IS2_atl03_geoid_attrs[gtx]['geolocation']['segment_id']['description'] = ("A 7 digit number "
            "identifying the along-track geolocation segment number.  These are sequential, starting with "
            "1 for the first segment after an ascending equatorial crossing node. Equal to the segment_id for "
            "the second of the two 20m ATL03 segments included in the 40m ATL03 segment")
        IS2_atl03_geoid_attrs[gtx]['geolocation']['segment_id']['coordinates'] = \
            "delta_time reference_photon_lat reference_photon_lon"

        #-- geoid undulation
        IS2_atl03_geoid[gtx]['geophys_corr']['geoid'] = N.astype(np.float64)
        IS2_atl03_fill[gtx]['geophys_corr']['geoid'] = None
        IS2_atl03_dims[gtx]['geophys_corr']['geoid'] = ['delta_time']
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid'] = {}
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['units'] = "meters"
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['contentType'] = "referenceInformation"
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['long_name'] = 'Geoidal_Undulation'
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['description'] = ('Geoidal '
            'undulation above the {0} ellipsoid').format(REFERENCE)
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['source'] = Ylms['modelname']
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['earth_gravity_constant'] = GM
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['radius'] = R
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['degree_of_truncation'] = LMAX
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid']['coordinates'] = \
            ("../geolocation/segment_id ../geolocation/delta_time "
            "../geolocation/reference_photon_lat ../geolocation/reference_photon_lon")
        #-- geoid conversion
        IS2_atl03_geoid[gtx]['geophys_corr']['geoid_free2mean'] = free2mean.copy()
        IS2_atl03_fill[gtx]['geophys_corr']['geoid_free2mean'] = None
        IS2_atl03_dims[gtx]['geophys_corr']['geoid_free2mean'] = ['delta_time']
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid_free2mean'] = {}
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid_free2mean']['units'] = "meters"
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid_free2mean']['contentType'] = "referenceInformation"
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid_free2mean']['long_name'] = ('Geoid_'
            'Free-to-Mean_conversion')
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid_free2mean']['description'] = ('Additive '
            'value to convert geoid heights from the tide-free system to the mean-tide system')
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid_free2mean']['earth_gravity_constant'] = GM
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid_free2mean']['radius'] = R
        IS2_atl03_geoid_attrs[gtx]['geophys_corr']['geoid_free2mean']['coordinates'] = \
            ("../geolocation/segment_id ../geolocation/delta_time "
            "../geolocation/reference_photon_lat ../geolocation/reference_photon_lon")

    #-- print file information
    logging.info('\t{0}'.format(OUTPUT_FILE))
    HDF5_ATL03_geoid_write(IS2_atl03_geoid, IS2_atl03_geoid_attrs,
        CLOBBER=True, INPUT=os.path.basename(INPUT_FILE),
        FILL_VALUE=IS2_atl03_fill, DIMENSIONS=IS2_atl03_dims,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE))
    #-- change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

#-- PURPOSE: outputting the geoid values for ICESat-2 data to HDF5
def HDF5_ATL03_geoid_write(IS2_atl03_geoid, IS2_atl03_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=False):
    #-- setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    #-- open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    #-- create HDF5 records
    h5 = {}

    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl03_geoid['ancillary_data'].items():
        #-- Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        #-- add HDF5 variable attributes
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    #-- write each output beam
    beams = [k for k in IS2_atl03_geoid.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        h5[gtx] = {}
        #-- add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl03_attrs[gtx][att_name]
        #-- create geolocation and geophys_corr groups
        for key in ['geolocation','geophys_corr']:
            fileID[gtx].create_group(key)
            h5[gtx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val = IS2_atl03_attrs[gtx][key][att_name]
                fileID[gtx][key].attrs[att_name] = att_val

            #-- all variables for group
            groupkeys = set(IS2_atl03_geoid[gtx][key].keys())-set(['delta_time'])
            for k in ['delta_time',*sorted(groupkeys)]:
                #-- values and attributes
                v = IS2_atl03_geoid[gtx][key][k]
                attrs = IS2_atl03_attrs[gtx][key][k]
                fillvalue = FILL_VALUE[gtx][key][k]
                #-- Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(gtx,key,k)
                if fillvalue:
                    h5[gtx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, fillvalue=fillvalue,
                        compression='gzip')
                else:
                    h5[gtx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, compression='gzip')
                #-- create or attach dimensions for HDF5 variable
                if DIMENSIONS[gtx][key][k]:
                    #-- attach dimensions
                    for i,dim in enumerate(DIMENSIONS[gtx][key][k]):
                        h5[gtx][key][k].dims[i].attach_scale(h5[gtx][key][dim])
                else:
                    #-- make dimension
                    h5[gtx][key][k].make_scale(k)
                #-- add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[gtx][key][k].attrs[att_name] = att_val

    #-- HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L2A Global Geolocated Photon Data'
    fileID.attrs['summary'] = ('The purpose of ATL03 is to provide along-track '
        'photon data for all 6 ATLAS beams and associated statistics')
    fileID.attrs['description'] = ('Photon heights determined by ATBD '
        'Algorithm using POD and PPD. All photon events per transmit pulse '
        'per beam. Includes POD and PPD vectors. Classification of each '
        'photon by several ATBD Algorithms.')
    date_created = datetime.datetime.today()
    fileID.attrs['date_created'] = date_created.isoformat()
    project = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = project
    platform = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = platform
    #-- add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    #-- add attributes for input ATL03 file
    fileID.attrs['input_files'] = os.path.basename(INPUT)
    #-- find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl03_geoid[gtx]['geolocation']['reference_photon_lon']
        lat = IS2_atl03_geoid[gtx]['geolocation']['reference_photon_lat']
        delta_time = IS2_atl03_geoid[gtx]['geolocation']['delta_time']
        #-- setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time.min() if (delta_time.min() < tmn) else tmn
        tmx = delta_time.max() if (delta_time.max() > tmx) else tmx
    #-- add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = ltmn
    fileID.attrs['geospatial_lat_max'] = ltmx
    fileID.attrs['geospatial_lon_min'] = lnmn
    fileID.attrs['geospatial_lon_max'] = lnmx
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['date_type'] = 'UTC'
    fileID.attrs['time_type'] = 'CCSDS UTC-A'
    #-- convert start and end time from ATLAS SDP seconds into UTC time
    time_utc = convert_delta_time(np.array([tmn,tmx]))
    #-- convert to calendar date
    YY,MM,DD,HH,MN,SS = icesat2_toolkit.time.convert_julian(time_utc['julian'],
        FORMAT='tuple')
    #-- add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = '{0:0.0f}'.format(tmx-tmn)
    #-- Closing the HDF5 file
    fileID.close()

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates tidal elevations for correcting ICESat-2 ATL03
            geolocated photon height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    #-- command line parameters
    #-- input ICESat-2 geolocated photon height files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL03 file to run')
    #-- set gravity model file to use
    parser.add_argument('--gravity','-G',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Gravity model file to use')
    #-- maximum spherical harmonic degree (level of truncation)
    parser.add_argument('--lmax','-l',
        type=int, help='Maximum spherical harmonic degree')
    #-- load love number of degree 2 (default EGM2008 value)
    parser.add_argument('--love','-n',
        type=float, default=0.3,
        help='Degree 2 load Love number')
    #-- verbosity settings
    #-- verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    #-- permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    #-- return the parser
    return parser

#-- This is the main part of the program that calls the individual functions
def main():
    #-- Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    #-- run for each input ATL03 file
    for FILE in args.infile:
        compute_geoid_ICESat2(args.gravity, FILE, LMAX=args.lmax,
            LOVE=args.love, VERBOSE=args.verbose, MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
