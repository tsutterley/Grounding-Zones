#!/usr/bin/env python
u"""
compute_LPET_ICESat2_ATL03.py
Written by Tyler Sutterley (05/2024)
Calculates long-period equilibrium tidal elevations for correcting ICESat-2
    geolocated photon height data
Will calculate the long-period tides for all ATL03 segments and not just ocean
    segments defined by the ocean tide mask
Calculated at ATL03 segment level using reference photon geolocation and time
Segment level corrections can be applied to the individual photon events (PEs)

COMMAND LINE OPTIONS:
    -O X, --output-directory X: input/output data directory
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
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/ATL03.py: reads ICESat-2 global geolocated photon data files
    utilities.py: download and management utilities for syncing files
    predict.py: calculates long-period equilibrium ocean tides

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 04/2024: use timescale for temporal operations
    Updated 08/2023: create s3 filesystem when using s3 urls as input
    Updated 05/2023: use timescale class for time conversion operations
        using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 07/2022: place some imports within try/except statements
    Updated 04/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 04/2021: can use a generically named ATL03 file as input
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: H5py deprecation warning change to use make_scale
        merged time conversion routines into module
    Written 11/2020
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import datetime
import numpy as np
import grounding_zones as gz

# attempt imports
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# PURPOSE: read ICESat-2 geolocated photon data (ATL03)
# compute long-period equilibrium tides at points and times
def compute_LPET_ICESat2(INPUT_FILE,
        OUTPUT_DIRECTORY=None,
        VERBOSE=False,
        MODE=0o775
    ):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd', level=loglevel)

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = \
            rx.findall(GRANULE).pop()
    except:
        # output long-period equilibrium tide HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_LPET{INPUT_FILE.suffix}'
    else:
        # output long-period equilibrium tide HDF5 file for ASAS/NSIDC granules
        args = (PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format = '{0}_LPET_{1}{2}{3}{4}{5}{6}_{7}{8}{9}_{10}_{11}{12}.h5'
        FILENAME = file_format.format(*args)
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # full path to output file
    OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(FILENAME)

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = is2tk.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = is2tk.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()
    # read data from input ATL03 file
    IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams = \
        is2tk.io.ATL03.read_main(INPUT_FILE, ATTRIBUTES=True)

    # copy variables for outputting to HDF5 file
    IS2_atl03_tide = {}
    IS2_atl03_fill = {}
    IS2_atl03_dims = {}
    IS2_atl03_tide_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl03_tide['ancillary_data'] = {}
    IS2_atl03_tide_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl03_tide['ancillary_data'][key] = IS2_atl03_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl03_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][key].items():
            IS2_atl03_tide_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        # output data dictionaries for beam
        IS2_atl03_tide[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_fill[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_dims[gtx] = dict(geolocation={}, geophys_corr={})
        IS2_atl03_tide_attrs[gtx] = dict(geolocation={}, geophys_corr={})

        # read data and attributes for beam
        val,attrs = is2tk.io.ATL03.read_beam(INPUT_FILE, gtx,
                                             ATTRIBUTES=True)
        # extract variables for computing equilibrium tides
        segment_id = val['geolocation']['segment_id'].copy()
        delta_time = val['geolocation']['delta_time'].copy()
        lon = val['geolocation']['reference_photon_lon'].copy()
        lat = val['geolocation']['reference_photon_lat'].copy()

        # create timescale from ATLAS Standard Epoch time
        # GPS seconds since 2018-01-01 00:00:00 UTC
        ts = timescale.time.Timescale().from_deltatime(delta_time,
            epoch=timescale.time._atlas_sdp_epoch, standard='GPS')
        tide_time = ts.tide + ts.tt_ut1

        # predict long-period equilibrium tides at latitudes and time
        tide_lpe = pyTMD.predict.equilibrium_tide(tide_time, lat)

        # group attributes for beam
        IS2_atl03_tide_attrs[gtx]['Description'] = attrs['Description']
        IS2_atl03_tide_attrs[gtx]['atlas_pce'] = attrs['atlas_pce']
        IS2_atl03_tide_attrs[gtx]['atlas_beam_type'] = attrs['atlas_beam_type']
        IS2_atl03_tide_attrs[gtx]['groundtrack_id'] = attrs['groundtrack_id']
        IS2_atl03_tide_attrs[gtx]['atmosphere_profile'] = attrs['atmosphere_profile']
        IS2_atl03_tide_attrs[gtx]['atlas_spot_number'] = attrs['atlas_spot_number']
        IS2_atl03_tide_attrs[gtx]['sc_orientation'] = attrs['sc_orientation']

        # group attributes for geolocation
        IS2_atl03_tide_attrs[gtx]['geolocation']['Description'] = ("Contains parameters related to "
            "geolocation.  The rate of all of these parameters is at the rate corresponding to the "
            "ICESat-2 Geolocation Along Track Segment interval (nominally 20 m along-track).")
        IS2_atl03_tide_attrs[gtx]['geolocation']['data_rate'] = ("Data within this group are "
            "stored at the ICESat-2 20m segment rate.")
        # group attributes for geophys_corr
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['Description'] = ("Contains parameters used to "
            "correct photon heights for geophysical effects, such as tides.  These parameters are "
            "posted at the same interval as the ICESat-2 Geolocation Along-Track Segment interval "
            "(nominally 20m along-track).")
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['data_rate'] = ("These parameters are stored at "
            "the ICESat-2 Geolocation Along Track Segment rate (nominally every 20 m along-track).")

        # geolocation, time and segment ID
        # delta time in geolocation group
        IS2_atl03_tide[gtx]['geolocation']['delta_time'] = delta_time
        IS2_atl03_fill[gtx]['geolocation']['delta_time'] = None
        IS2_atl03_dims[gtx]['geolocation']['delta_time'] = None
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time'] = {}
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['standard_name'] = "time"
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['calendar'] = "standard"
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['description'] = ("Elapsed seconds "
            "from the ATLAS SDP GPS Epoch, corresponding to the transmit time of the reference "
            "photon. The ATLAS Standard Data Products (SDP) epoch offset is defined within "
            "/ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds between the GPS epoch "
            "(1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By adding the offset "
            "contained within atlas_sdp_gps_epoch to delta time parameters, the time in gps_seconds "
            "relative to the GPS epoch can be computed.")
        IS2_atl03_tide_attrs[gtx]['geolocation']['delta_time']['coordinates'] = \
            "segment_id reference_photon_lat reference_photon_lon"
        # delta time in geophys_corr group
        IS2_atl03_tide[gtx]['geophys_corr']['delta_time'] = delta_time
        IS2_atl03_fill[gtx]['geophys_corr']['delta_time'] = None
        IS2_atl03_dims[gtx]['geophys_corr']['delta_time'] = None
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time'] = {}
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['standard_name'] = "time"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['calendar'] = "standard"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['description'] = ("Elapsed seconds "
            "from the ATLAS SDP GPS Epoch, corresponding to the transmit time of the reference "
            "photon. The ATLAS Standard Data Products (SDP) epoch offset is defined within "
            "/ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds between the GPS epoch "
            "(1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By adding the offset "
            "contained within atlas_sdp_gps_epoch to delta time parameters, the time in gps_seconds "
            "relative to the GPS epoch can be computed.")
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['delta_time']['coordinates'] = ("../geolocation/segment_id "
            "../geolocation/reference_photon_lat ../geolocation/reference_photon_lon")

        # latitude
        IS2_atl03_tide[gtx]['geolocation']['reference_photon_lat'] = lat
        IS2_atl03_fill[gtx]['geolocation']['reference_photon_lat'] = None
        IS2_atl03_dims[gtx]['geolocation']['reference_photon_lat'] = ['delta_time']
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat'] = {}
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['units'] = "degrees_north"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['contentType'] = "physicalMeasurement"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['long_name'] = "Latitude"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['standard_name'] = "latitude"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['description'] = ("Latitude of each "
            "reference photon. Computed from the ECF Cartesian coordinates of the bounce point.")
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['valid_min'] = -90.0
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['valid_max'] = 90.0
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lat']['coordinates'] = \
            "segment_id delta_time reference_photon_lon"
        # longitude
        IS2_atl03_tide[gtx]['geolocation']['reference_photon_lon'] = lon
        IS2_atl03_fill[gtx]['geolocation']['reference_photon_lon'] = None
        IS2_atl03_dims[gtx]['geolocation']['reference_photon_lon'] = ['delta_time']
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon'] = {}
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['units'] = "degrees_east"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['contentType'] = "physicalMeasurement"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['long_name'] = "Longitude"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['standard_name'] = "longitude"
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['description'] = ("Longitude of each "
            "reference photon. Computed from the ECF Cartesian coordinates of the bounce point.")
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['valid_min'] = -180.0
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['valid_max'] = 180.0
        IS2_atl03_tide_attrs[gtx]['geolocation']['reference_photon_lon']['coordinates'] = \
            "segment_id delta_time reference_photon_lat"
        # segment ID
        IS2_atl03_tide[gtx]['geolocation']['segment_id'] = segment_id
        IS2_atl03_fill[gtx]['geolocation']['segment_id'] = None
        IS2_atl03_dims[gtx]['geolocation']['segment_id'] = ['delta_time']
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id'] = {}
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['units'] = "1"
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['contentType'] = "referenceInformation"
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['long_name'] = "Along-track segment ID number"
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['description'] = ("A 7 digit number "
            "identifying the along-track geolocation segment number.  These are sequential, starting with "
            "1 for the first segment after an ascending equatorial crossing node.")
        IS2_atl03_tide_attrs[gtx]['geolocation']['segment_id']['coordinates'] = \
            "delta_time reference_photon_lat reference_photon_lon"

        # computed long-period equilibrium tide
        IS2_atl03_tide[gtx]['geophys_corr']['tide_equilibrium'] = tide_lpe
        IS2_atl03_fill[gtx]['geophys_corr']['tide_equilibrium'] = None
        IS2_atl03_dims[gtx]['geophys_corr']['tide_equilibrium'] = ['delta_time']
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['tide_equilibrium'] = {}
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['tide_equilibrium']['units'] = "meters"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['tide_equilibrium']['contentType'] = "referenceInformation"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['tide_equilibrium']['long_name'] = \
            "Long Period Equilibrium Tide"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['tide_equilibrium']['description'] = ("Long-period "
            "equilibrium tidal elevation from the summation of fifteen tidal spectral lines")
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['tide_equilibrium']['reference'] = \
            "https://doi.org/10.1111/j.1365-246X.1973.tb03420.x"
        IS2_atl03_tide_attrs[gtx]['geophys_corr']['tide_equilibrium']['coordinates'] = \
            ("../geolocation/segment_id ../geolocation/delta_time "
            "../geolocation/reference_photon_lat ../geolocation/reference_photon_lon")

    # print file information
    logger.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_ATL03_tide_write(IS2_atl03_tide, IS2_atl03_tide_attrs,
        FILENAME=OUTPUT_FILE,
        INPUT=GRANULE,
        FILL_VALUE=IS2_atl03_fill,
        DIMENSIONS=IS2_atl03_dims,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the tide values for ICESat-2 data to HDF5
def HDF5_ATL03_tide_write(IS2_atl03_tide, IS2_atl03_attrs, INPUT=None,
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
    for k,v in IS2_atl03_tide['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl03_tide.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        h5[gtx] = {}
        # add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl03_attrs[gtx][att_name]
        # create geolocation and geophys_corr groups
        for key in ['geolocation','geophys_corr']:
            fileID[gtx].create_group(key)
            h5[gtx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val = IS2_atl03_attrs[gtx][key][att_name]
                fileID[gtx][key].attrs[att_name] = att_val

            # all variables for group
            groupkeys = set(IS2_atl03_tide[gtx][key].keys())-set(['delta_time'])
            for k in ['delta_time',*sorted(groupkeys)]:
                # values and attributes
                v = IS2_atl03_tide[gtx][key][k]
                attrs = IS2_atl03_attrs[gtx][key][k]
                fillvalue = FILL_VALUE[gtx][key][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(gtx,key,k)
                if fillvalue:
                    h5[gtx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, fillvalue=fillvalue,
                        compression='gzip')
                else:
                    h5[gtx][key][k] = fileID.create_dataset(val, np.shape(v),
                        data=v, dtype=v.dtype, compression='gzip')
                # create or attach dimensions for HDF5 variable
                if DIMENSIONS[gtx][key][k]:
                    # attach dimensions
                    for i,dim in enumerate(DIMENSIONS[gtx][key][k]):
                        h5[gtx][key][k].dims[i].attach_scale(h5[gtx][key][dim])
                else:
                    # make dimension
                    h5[gtx][key][k].make_scale(k)
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[gtx][key][k].attrs[att_name] = att_val

    # HDF5 file title
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
    # add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    # add attributes for input ATL03 file
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl03_tide[gtx]['geolocation']['reference_photon_lon']
        lat = IS2_atl03_tide[gtx]['geolocation']['reference_photon_lat']
        delta_time = IS2_atl03_tide[gtx]['geolocation']['delta_time']
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
    fileID.attrs['software_reference'] = pyTMD.version.project_name
    fileID.attrs['software_version'] = pyTMD.version.full_version
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates long-period equilibrium tidal elevations for
            correcting ICESat-2 ATL03 geolocated photon height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL03 file to run')
    # directory with output data
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

    # run for each input ATL03 file
    for FILE in args.infile:
        compute_LPET_ICESat2(FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
