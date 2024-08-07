#!/usr/bin/env python
u"""
compute_LPET_ICESat2_ATL06.py
Written by Tyler Sutterley (05/2024)
Calculates long-period equilibrium tidal elevations for correcting ICESat-2
    land ice elevation data
Will calculate the long-period tides for all ATL06 segments and not just ocean
    segments defined by the ocean tide mask

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
    io/ATL06.py: reads ICESat-2 land ice along-track height data files
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
    Updated 04/2021: can use a generically named ATL06 file as input
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

# PURPOSE: read ICESat-2 land ice data (ATL06)
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
    # read data from input file
    IS2_atl06_mds,IS2_atl06_attrs,IS2_atl06_beams = \
        is2tk.io.ATL06.read_granule(INPUT_FILE, ATTRIBUTES=True)

    # copy variables for outputting to HDF5 file
    IS2_atl06_tide = {}
    IS2_atl06_fill = {}
    IS2_atl06_dims = {}
    IS2_atl06_tide_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl06_tide['ancillary_data'] = {}
    IS2_atl06_tide_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl06_tide['ancillary_data'][key] = IS2_atl06_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl06_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][key].items():
            IS2_atl06_tide_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl06_beams):
        # output data dictionaries for beam
        IS2_atl06_tide[gtx] = dict(land_ice_segments={})
        IS2_atl06_fill[gtx] = dict(land_ice_segments={})
        IS2_atl06_dims[gtx] = dict(land_ice_segments={})
        IS2_atl06_tide_attrs[gtx] = dict(land_ice_segments={})

        # number of segments
        val = IS2_atl06_mds[gtx]['land_ice_segments']
        n_seg = len(val['segment_id'])
        # find valid segments for beam
        fv = IS2_atl06_attrs[gtx]['land_ice_segments']['h_li']['_FillValue']

        # create timescale from ATLAS Standard Epoch time
        # GPS seconds since 2018-01-01 00:00:00 UTC
        ts = timescale.time.Timescale().from_deltatime(val['delta_time'],
            epoch=timescale.time._atlas_sdp_epoch, standard='GPS')
        tide_time = ts.tide + ts.tt_ut1

        # predict long-period equilibrium tides at latitudes and time
        tide_lpe = np.ma.zeros((n_seg), fill_value=fv)
        tide_lpe.data[:] = pyTMD.predict.equilibrium_tide(tide_time, val['latitude'])
        tide_lpe.mask = (val['latitude'] == fv) | (val['delta_time'] == fv)

        # group attributes for beam
        IS2_atl06_tide_attrs[gtx]['Description'] = IS2_atl06_attrs[gtx]['Description']
        IS2_atl06_tide_attrs[gtx]['atlas_pce'] = IS2_atl06_attrs[gtx]['atlas_pce']
        IS2_atl06_tide_attrs[gtx]['atlas_beam_type'] = IS2_atl06_attrs[gtx]['atlas_beam_type']
        IS2_atl06_tide_attrs[gtx]['groundtrack_id'] = IS2_atl06_attrs[gtx]['groundtrack_id']
        IS2_atl06_tide_attrs[gtx]['atmosphere_profile'] = IS2_atl06_attrs[gtx]['atmosphere_profile']
        IS2_atl06_tide_attrs[gtx]['atlas_spot_number'] = IS2_atl06_attrs[gtx]['atlas_spot_number']
        IS2_atl06_tide_attrs[gtx]['sc_orientation'] = IS2_atl06_attrs[gtx]['sc_orientation']
        # group attributes for land_ice_segments
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['Description'] = ("The land_ice_segments group "
            "contains the primary set of derived products. This includes geolocation, height, and "
            "standard error and quality measures for each segment. This group is sparse, meaning "
            "that parameters are provided only for pairs of segments for which at least one beam "
            "has a valid surface-height measurement.")
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['data_rate'] = ("Data within this group are "
            "sparse.  Data values are provided only for those ICESat-2 20m segments where at "
            "least one beam has a valid land ice height measurement.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl06_tide[gtx]['land_ice_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['delta_time'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['delta_time'] = None
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['delta_time'] = {}
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['delta_time']['standard_name'] = "time"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['delta_time']['calendar'] = "standard"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['delta_time']['coordinates'] = \
            "segment_id latitude longitude"
        # latitude
        IS2_atl06_tide[gtx]['land_ice_segments']['latitude'] = val['latitude'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['latitude'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['latitude'] = ['delta_time']
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude'] = {}
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude']['units'] = "degrees_north"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude']['valid_min'] = -90.0
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude']['valid_max'] = 90.0
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['latitude']['coordinates'] = \
            "segment_id delta_time longitude"
        # longitude
        IS2_atl06_tide[gtx]['land_ice_segments']['longitude'] = val['longitude'].copy()
        IS2_atl06_fill[gtx]['land_ice_segments']['longitude'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['longitude'] = ['delta_time']
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude'] = {}
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude']['units'] = "degrees_east"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude']['valid_min'] = -180.0
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude']['valid_max'] = 180.0
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['longitude']['coordinates'] = \
            "segment_id delta_time latitude"
        # segment ID
        IS2_atl06_tide[gtx]['land_ice_segments']['segment_id'] = val['segment_id']
        IS2_atl06_fill[gtx]['land_ice_segments']['segment_id'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['segment_id'] = ['delta_time']
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['segment_id'] = {}
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['segment_id']['units'] = "1"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['segment_id']['contentType'] = "referenceInformation"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['segment_id']['long_name'] = "Along-track segment ID number"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['segment_id']['description'] = ("A 7 digit number "
            "identifying the along-track geolocation segment number.  These are sequential, starting with "
            "1 for the first segment after an ascending equatorial crossing node. Equal to the segment_id for "
            "the second of the two 20m ATL03 segments included in the 40m ATL06 segment")
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['segment_id']['coordinates'] = \
            "delta_time latitude longitude"

        # geophysical variables
        IS2_atl06_tide[gtx]['land_ice_segments']['geophysical'] = {}
        IS2_atl06_fill[gtx]['land_ice_segments']['geophysical'] = {}
        IS2_atl06_dims[gtx]['land_ice_segments']['geophysical'] = {}
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical'] = {}
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['Description'] = ("The geophysical group "
            "contains parameters used to correct segment heights for geophysical effects, parameters "
            "related to solar background and parameters indicative of the presence or absence of clouds.")
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['data_rate'] = ("Data within this group "
            "are stored at the land_ice_segments segment rate.")
        # computed long-period equilibrium tide
        IS2_atl06_tide[gtx]['land_ice_segments']['geophysical']['tide_equilibrium'] = tide_lpe
        IS2_atl06_fill[gtx]['land_ice_segments']['geophysical']['tide_equilibrium'] = tide_lpe.fill_value
        IS2_atl06_dims[gtx]['land_ice_segments']['geophysical']['tide_equilibrium'] = ['delta_time']
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['tide_equilibrium'] = {}
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['tide_equilibrium']['units'] = "meters"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['tide_equilibrium']['contentType'] = "referenceInformation"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['tide_equilibrium']['long_name'] = \
            "Long Period Equilibrium Tide"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['tide_equilibrium']['description'] = ("Long-period "
            "equilibrium tidal elevation from the summation of fifteen tidal spectral lines")
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['tide_equilibrium']['reference'] = \
            "https://doi.org/10.1111/j.1365-246X.1973.tb03420.x"
        IS2_atl06_tide_attrs[gtx]['land_ice_segments']['geophysical']['tide_equilibrium']['coordinates'] = \
            "../segment_id ../delta_time ../latitude ../longitude"

    # print file information
    logger.info(f'\t{str(OUTPUT_FILE)}')
    HDF5_ATL06_tide_write(IS2_atl06_tide, IS2_atl06_tide_attrs,
        FILENAME=OUTPUT_FILE,
        INPUT=GRANULE,
        FILL_VALUE=IS2_atl06_fill,
        DIMENSIONS=IS2_atl06_dims,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

# PURPOSE: outputting the tide values for ICESat-2 data to HDF5
def HDF5_ATL06_tide_write(IS2_atl06_tide, IS2_atl06_attrs, INPUT=None,
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
    for k,v in IS2_atl06_tide['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl06_tide.keys() if bool(re.match(r'gt\d[lr]',k))]
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

        # delta_time, geolocation and segment_id variables
        for k in ['delta_time','latitude','longitude','segment_id']:
            # values and attributes
            v = IS2_atl06_tide[gtx]['land_ice_segments'][k]
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

        # add to geophysical corrections
        key = 'geophysical'
        fileID[gtx]['land_ice_segments'].create_group(key)
        h5[gtx]['land_ice_segments'][key] = {}
        for att_name in ['Description','data_rate']:
            att_val=IS2_atl06_attrs[gtx]['land_ice_segments'][key][att_name]
            fileID[gtx]['land_ice_segments'][key].attrs[att_name] = att_val
        for k,v in IS2_atl06_tide[gtx]['land_ice_segments'][key].items():
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
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L3A Land Ice Height'
    fileID.attrs['summary'] = ('Estimates of the ice-sheet tidal parameters '
        'needed to interpret and assess the quality of the height estimates.')
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
    # add attributes for input ATL06 file
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl06_tide[gtx]['land_ice_segments']['longitude']
        lat = IS2_atl06_tide[gtx]['land_ice_segments']['latitude']
        delta_time = IS2_atl06_tide[gtx]['land_ice_segments']['delta_time']
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
            correcting ICESat-2 ATL06 land ice elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL06 file to run')
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

    # run for each input ATL06 file
    for FILE in args.infile:
        compute_LPET_ICESat2(FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
