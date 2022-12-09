#!/usr/bin/env python
u"""
adjust_tides_ICESat2_ATL11.py
Written by Tyler Sutterley (12/2022)
Applies interpolated tidal adjustment scale factors to
    ICESat-2 ATL11 annual land ice height data within
    ice sheet grounding zones

COMMAND LINE OPTIONS:
    --help: list the command line options
    -f X, --flexure-file X: Ice flexure file to use
    -T X, --tide X: Tide model to use in correction
    -V, --verbose: Verbose output of run
    -m X, --mode X: Local permissions mode of the output mosaic

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/

PROGRAM DEPENDENCIES:
    model.py: retrieves tide model parameters for named tide models
    spatial.py: utilities for reading and writing spatial data
    read_ICESat2_ATL11.py: reads ICESat-2 annual land ice height data files

UPDATE HISTORY:
    Updated 12/2022: single implicit import of grounding zone tools
    Updated 07/2022: place some imports within try/except statements
    Written 06/2022
"""

import os
import re
import pyproj
import logging
import argparse
import datetime
import warnings
import collections
import numpy as np
import scipy.interpolate

# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("h5py not available")
try:
    import icesat2_toolkit as is2tk
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("icesat2_toolkit not available")
try:
    import pyTMD
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("pyTMD not available")
# ignore warnings
warnings.filterwarnings("ignore")

def adjust_tides_ICESat2_ATL11(adjustment_file, INPUT_FILE,
    TIDE_MODEL=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd',level=loglevel)

    # get tide model parameters
    model = pyTMD.model(None, verify=False).elevation(TIDE_MODEL)
    # source of tide model
    tide_source = TIDE_MODEL
    tide_reference = model.reference

    # read data from input file
    logger.info(f'{INPUT_FILE} -->')
    IS2_atl11_mds,IS2_atl11_attrs,IS2_atl11_pairs = \
        is2tk.read_HDF5_ATL11(INPUT_FILE, ATTRIBUTES=True, CROSSOVERS=True)
    DIRECTORY = os.path.dirname(INPUT_FILE)
    # flexure flag if being applied
    flexure_flag = '_FLEXURE'
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(INPUT_FILE).pop()
    except:
        # output tide HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        args = (fileBasename,model.name,flexure_flag,fileExtension)
        OUTPUT_FILE = '{0}_{1}{2}_TIDES{3}'.format(*args)
    else:
        # output tide HDF5 file for ASAS/NSIDC granules
        args = (PRD,model.name,flexure_flag,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        file_format = '{0}_{1}{2}_TIDES_{3}{4}_{5}{6}_{7}_{8}{9}.h5'
        OUTPUT_FILE = file_format.format(*args)

    # read tide adjustment file
    adjustment = pyTMD.spatial.from_HDF5(adjustment_file,
        field_mapping=dict(x='x', y='y', data='tide_adj_scale'))
    # tide adjustment grid spacing
    dx = np.abs(adjustment['x'][1] - adjustment['x'][0])
    dy = np.abs(adjustment['y'][1] - adjustment['y'][0])

    # create coordinate reference systems for converting to projection
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_wkt(adjustment['attributes']['crs']['crs_wkt'])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # read orbit info for bounding polygons
    bounding_lon = IS2_atl11_mds['orbit_info']['bounding_polygon_lon1']
    bounding_lat = IS2_atl11_mds['orbit_info']['bounding_polygon_lat1']
    # convert bounding polygon coordinates to projection
    BX,BY = transformer.transform(bounding_lon, bounding_lat)
    # determine bounds of data in image coordinates (affine transform)
    xmin = (BX.min() - adjustment['x'][0])//dx
    xmax = (BX.max() - adjustment['x'][0])//dx
    ymin = (BY.min() - adjustment['y'][0])//dy
    ymax = (BY.max() - adjustment['y'][0])//dy
    # reduce tide adjustment grid to buffered bounds of data
    # and convert invalid values to 0
    indx = slice(int(xmin)-10, int(xmax)+10, 1)
    indy = slice(int(ymin)-10, int(ymax)+10, 1)
    tide_adj_scale = np.nan_to_num(adjustment['data'][indy,indx], nan=0.0)
    # create interpolator for tide adjustment
    SPL = scipy.interpolate.RectBivariateSpline(
        adjustment['x'][indx], adjustment['y'][indy],
        tide_adj_scale.T, kx=1, ky=1)

    # copy variables for outputting to HDF5 file
    IS2_atl11_tide = {}
    IS2_atl11_fill = {}
    IS2_atl11_dims = {}
    IS2_atl11_tide_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl11_tide['ancillary_data'] = {}
    IS2_atl11_tide_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl11_tide['ancillary_data'][key] = IS2_atl11_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl11_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][key].items():
            IS2_atl11_tide_attrs['ancillary_data'][key][att_name] = att_val
    # HDF5 group name for across-track data
    XT = 'crossing_track_data'

    # for each input beam pair within the file
    for ptx in sorted(IS2_atl11_pairs):
        # output data dictionaries for beam
        IS2_atl11_tide[ptx] = dict(cycle_stats=collections.OrderedDict(),
            crossing_track_data=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(cycle_stats={},crossing_track_data={})
        IS2_atl11_dims[ptx] = dict(cycle_stats={},crossing_track_data={})
        IS2_atl11_tide_attrs[ptx] = dict(cycle_stats={},crossing_track_data={})

        # extract along-track and across-track variables
        ref_pt = {}
        latitude = {}
        longitude = {}
        delta_time = {}
        # cycle numbers
        cycle_number = IS2_atl11_mds[ptx]['cycle_number'].copy()
        # along-track (AT) reference point, latitude, longitude and time
        ref_pt['AT'] = IS2_atl11_mds[ptx]['ref_pt'].copy()
        latitude['AT'] = np.ma.array(IS2_atl11_mds[ptx]['latitude'],
            fill_value=IS2_atl11_attrs[ptx]['latitude']['_FillValue'])
        longitude['AT'] = np.ma.array(IS2_atl11_mds[ptx]['longitude'],
            fill_value=IS2_atl11_attrs[ptx]['longitude']['_FillValue'])
        delta_time['AT'] = np.ma.array(IS2_atl11_mds[ptx]['delta_time'],
            fill_value=IS2_atl11_attrs[ptx]['delta_time']['_FillValue'])
        # across-track (XT) reference point, latitude, longitude and time
        ref_pt['XT'] = IS2_atl11_mds[ptx][XT]['ref_pt'].copy()
        latitude['XT'] = np.ma.array(IS2_atl11_mds[ptx][XT]['latitude'],
            fill_value=IS2_atl11_attrs[ptx][XT]['latitude']['_FillValue'])
        longitude['XT'] = np.ma.array(IS2_atl11_mds[ptx][XT]['longitude'],
            fill_value=IS2_atl11_attrs[ptx][XT]['longitude']['_FillValue'])
        delta_time['XT'] = np.ma.array(IS2_atl11_mds[ptx][XT]['delta_time'],
            fill_value=IS2_atl11_attrs[ptx][XT]['delta_time']['_FillValue'])

        # number of average segments and number of included cycles
        # fill_value for invalid heights and corrections
        fv = IS2_atl11_attrs[ptx]['h_corr']['_FillValue']
        # shape of along-track and across-track data
        n_points,n_cycles = delta_time['AT'].shape
        n_cross, = delta_time['XT'].shape
        # allocate for output tidal variables
        tide = {}
        tide_adj = {}
        # along-track (AT) tides and tidal adjustments
        tide['AT'] = np.ma.empty((n_points,n_cycles),fill_value=fv)
        tide['AT'].mask = (delta_time['AT'] == delta_time['AT'].fill_value)
        tide_adj['AT'] = np.ma.empty((n_points),fill_value=fv)
        tide_adj['AT'].mask = np.ones((n_points), dtype=bool)
        # across-track (XT) tides and tidal adjustments
        tide['XT'] = np.ma.empty((n_cross),fill_value=fv)
        tide['XT'].mask = (delta_time['XT'] == delta_time['XT'].fill_value)
        tide_adj['XT'] = np.ma.empty((n_cross),fill_value=fv)
        tide_adj['XT'].mask = np.ones((n_cross), dtype=bool)

        # read tide model HDF5 file
        a3 = (PRD,TIDE_MODEL,'',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        f3 = os.path.join(DIRECTORY,file_format.format(*a3))
        print(f3)
        # check that tide file exists
        try:
            mds3,attr3 = is2tk.read_HDF5_ATL11_pair(f3, ptx,
                VERBOSE=False,CROSSOVERS=True)
        except:
            # mask all values
            for group in ['AT','XT']:
                tide[group].mask[:] = True
            pass
        else:
            # read tide model data
            tide['AT'].data[:] = mds3[ptx]['cycle_stats']['tide_ocean']
            tide['XT'].data[:] = mds3[ptx][XT]['tide_ocean']
            # update tide model masks
            tide['AT'].mask[:] |= (tide['AT'].data == tide['AT'].fill_value)
            tide['XT'].mask[:] |= (tide['XT'].data == tide['XT'].fill_value)

        # adjust tides for along-track and across-track data
        for track in ['AT','XT']:
            # find valid points
            ii, = np.nonzero((latitude[track] != latitude[track].fill_value) |
                (longitude[track] != longitude[track].fill_value))
            # convert coordinates to projection
            X,Y = transformer.transform(longitude[track][ii], latitude[track][ii])
            # interpolate tide adjustment scale
            tide_adj[track].data[ii] = SPL.ev(X, Y)
            tide_adj[track].mask[ii] = False
            # apply flexure scaling factors
            if (track == 'AT'):
                # for each along-track cycle
                for c,cycle in enumerate(cycle_number):
                    # print('b',cycle,tide[track].data[ii,c])
                    tide[track].data[ii,c] *= tide_adj[track][ii]
                    # print('a',cycle,tide[track].data[ii,c])
            elif (track == 'XT'):
                # for all across-track points
                tide[track].data[ii] *= tide_adj[track][ii]
            # replace masked and nan values with fill value
            invalid = np.nonzero(np.isnan(tide[track].data) | tide[track].mask)
            tide[track].data[invalid] = tide[track].fill_value
            tide[track].mask[invalid] = True
            # replace adjustment masked points with fill value
            tide_adj[track].data[tide_adj[track].mask] = tide_adj[track].fill_value

        # group attributes for beam
        IS2_atl11_tide_attrs[ptx]['description'] = ('Contains the primary '
            'science parameters for this data set')
        IS2_atl11_tide_attrs[ptx]['beam_pair'] = IS2_atl11_attrs[ptx]['beam_pair']
        IS2_atl11_tide_attrs[ptx]['ReferenceGroundTrack'] = IS2_atl11_attrs[ptx]['ReferenceGroundTrack']
        IS2_atl11_tide_attrs[ptx]['first_cycle'] = IS2_atl11_attrs[ptx]['first_cycle']
        IS2_atl11_tide_attrs[ptx]['last_cycle'] = IS2_atl11_attrs[ptx]['last_cycle']
        IS2_atl11_tide_attrs[ptx]['equatorial_radius'] = IS2_atl11_attrs[ptx]['equatorial_radius']
        IS2_atl11_tide_attrs[ptx]['polar_radius'] = IS2_atl11_attrs[ptx]['polar_radius']

        # geolocation, time and reference point
        # reference point
        IS2_atl11_tide[ptx]['ref_pt'] = ref_pt['AT'].copy()
        IS2_atl11_fill[ptx]['ref_pt'] = None
        IS2_atl11_dims[ptx]['ref_pt'] = None
        IS2_atl11_tide_attrs[ptx]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['ref_pt']['units'] = "1"
        IS2_atl11_tide_attrs[ptx]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['ref_pt']['long_name'] = "Reference point number"
        IS2_atl11_tide_attrs[ptx]['ref_pt']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['ref_pt']['description'] = ("The reference point is the "
            "7 digit segment_id number corresponding to the center of the ATL06 data used "
            "for each ATL11 point.  These are sequential, starting with 1 for the first "
            "segment after an ascending equatorial crossing node.")
        IS2_atl11_tide_attrs[ptx]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number
        IS2_atl11_tide[ptx]['cycle_number'] = np.copy(cycle_number)
        IS2_atl11_fill[ptx]['cycle_number'] = None
        IS2_atl11_dims[ptx]['cycle_number'] = None
        IS2_atl11_tide_attrs[ptx]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_number']['units'] = "1"
        IS2_atl11_tide_attrs[ptx]['cycle_number']['long_name'] = "Orbital cycle number"
        IS2_atl11_tide_attrs[ptx]['cycle_number']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['cycle_number']['description'] = ("Number of 91-day periods "
            "that have elapsed since ICESat-2 entered the science orbit. Each of the 1,387 "
            "reference ground track (RGTs) is targeted in the polar regions once "
            "every 91 days.")
        # delta time
        IS2_atl11_tide[ptx]['delta_time'] = delta_time['AT'].copy()
        IS2_atl11_fill[ptx]['delta_time'] = delta_time['AT'].fill_value
        IS2_atl11_dims[ptx]['delta_time'] = ['ref_pt','cycle_number']
        IS2_atl11_tide_attrs[ptx]['delta_time'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_tide_attrs[ptx]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_tide_attrs[ptx]['delta_time']['standard_name'] = "time"
        IS2_atl11_tide_attrs[ptx]['delta_time']['calendar'] = "standard"
        IS2_atl11_tide_attrs[ptx]['delta_time']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_tide_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt cycle_number latitude longitude"
        # latitude
        IS2_atl11_tide[ptx]['latitude'] = latitude['AT'].copy()
        IS2_atl11_fill[ptx]['latitude'] = latitude['AT'].fill_value
        IS2_atl11_dims[ptx]['latitude'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx]['latitude'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['latitude']['units'] = "degrees_north"
        IS2_atl11_tide_attrs[ptx]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_tide_attrs[ptx]['latitude']['long_name'] = "Latitude"
        IS2_atl11_tide_attrs[ptx]['latitude']['standard_name'] = "latitude"
        IS2_atl11_tide_attrs[ptx]['latitude']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_tide_attrs[ptx]['latitude']['valid_min'] = -90.0
        IS2_atl11_tide_attrs[ptx]['latitude']['valid_max'] = 90.0
        IS2_atl11_tide_attrs[ptx]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude
        IS2_atl11_tide[ptx]['longitude'] = longitude['AT'].copy()
        IS2_atl11_fill[ptx]['longitude'] = longitude['AT'].fill_value
        IS2_atl11_dims[ptx]['longitude'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx]['longitude'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['longitude']['units'] = "degrees_east"
        IS2_atl11_tide_attrs[ptx]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_tide_attrs[ptx]['longitude']['long_name'] = "Longitude"
        IS2_atl11_tide_attrs[ptx]['longitude']['standard_name'] = "longitude"
        IS2_atl11_tide_attrs[ptx]['longitude']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_tide_attrs[ptx]['longitude']['valid_min'] = -180.0
        IS2_atl11_tide_attrs[ptx]['longitude']['valid_max'] = 180.0
        IS2_atl11_tide_attrs[ptx]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"

        # cycle statistics variables
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['Description'] = ("The cycle_stats subgroup "
            "contains summary information about segments for each reference point, including "
            "the uncorrected mean heights for reference surfaces, blowing snow and cloud "
            "indicators, and geolocation and height misfit statistics.")
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['data_rate'] = ("Data within this group "
            "are stored at the average segment rate.")
        # computed tide
        IS2_atl11_tide[ptx]['cycle_stats'][model.atl11] = tide['AT'].copy()
        IS2_atl11_fill[ptx]['cycle_stats'][model.atl11] = tide['AT'].fill_value
        IS2_atl11_dims[ptx]['cycle_stats'][model.atl11] = ['ref_pt','cycle_number']
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['units'] = "meters"
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['long_name'] = model.long_name
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['description'] = model.description
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['source'] = model.name
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['reference'] = model.reference
        IS2_atl11_tide_attrs[ptx]['cycle_stats'][model.atl11]['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"
        # computed tide adjustments
        IS2_atl11_tide[ptx]['cycle_stats']['tide_adj'] = tide_adj['AT'].copy()
        IS2_atl11_fill[ptx]['cycle_stats']['tide_adj'] = tide_adj['AT'].fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['tide_adj'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['units'] = "1"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['long_name'] = "Ocean Tide Adjustment"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['description'] = ("Interpolated "
            "empirical adjustment applied to the ocean tides for ice flexure.")
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['source'] = tide_source
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['reference'] = tide_reference
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['coordinates'] = \
            "../ref_pt ../latitude ../longitude"

        # crossing track variables
        IS2_atl11_tide_attrs[ptx][XT]['Description'] = ("The crossing_track_data "
            "subgroup contains elevation data at crossover locations. These are "
            "locations where two ICESat-2 pair tracks cross, so data are available "
            "from both the datum track, for which the granule was generated, and "
            "from the crossing track.")
        IS2_atl11_tide_attrs[ptx][XT]['data_rate'] = ("Data within this group are "
            "stored at the average segment rate.")

        # reference point
        IS2_atl11_tide[ptx][XT]['ref_pt'] = ref_pt['XT'].copy()
        IS2_atl11_fill[ptx][XT]['ref_pt'] = None
        IS2_atl11_dims[ptx][XT]['ref_pt'] = None
        IS2_atl11_tide_attrs[ptx][XT]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['units'] = "1"
        IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['long_name'] = ("fit center reference point number, "
            "segment_id")
        IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['source'] = "derived, ATL11 algorithm"
        IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['description'] = ("The reference-point number of the "
            "fit center for the datum track. The reference point is the 7 digit segment_id number "
            "corresponding to the center of the ATL06 data used for each ATL11 point.  These are "
            "sequential, starting with 1 for the first segment after an ascending equatorial "
            "crossing node.")
        IS2_atl11_tide_attrs[ptx][XT]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"

        # reference ground track of the crossing track
        IS2_atl11_tide[ptx][XT]['rgt'] = IS2_atl11_mds[ptx][XT]['rgt'].copy()
        IS2_atl11_fill[ptx][XT]['rgt'] = IS2_atl11_attrs[ptx][XT]['rgt']['_FillValue']
        IS2_atl11_dims[ptx][XT]['rgt'] = None
        IS2_atl11_tide_attrs[ptx][XT]['rgt'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['rgt']['units'] = "1"
        IS2_atl11_tide_attrs[ptx][XT]['rgt']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx][XT]['rgt']['long_name'] = "crossover reference ground track"
        IS2_atl11_tide_attrs[ptx][XT]['rgt']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx][XT]['rgt']['description'] = "The RGT number for the crossing data."
        IS2_atl11_tide_attrs[ptx][XT]['rgt']['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # cycle_number of the crossing track
        IS2_atl11_tide[ptx][XT]['cycle_number'] = IS2_atl11_mds[ptx][XT]['cycle_number'].copy()
        IS2_atl11_fill[ptx][XT]['cycle_number'] = IS2_atl11_attrs[ptx][XT]['cycle_number']['_FillValue']
        IS2_atl11_dims[ptx][XT]['cycle_number'] = None
        IS2_atl11_tide_attrs[ptx][XT]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['cycle_number']['units'] = "1"
        IS2_atl11_tide_attrs[ptx][XT]['cycle_number']['long_name'] = "crossover cycle number"
        IS2_atl11_tide_attrs[ptx][XT]['cycle_number']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx][XT]['cycle_number']['description'] = ("Cycle number for the "
            "crossing data. Number of 91-day periods that have elapsed since ICESat-2 entered "
            "the science orbit. Each of the 1,387 reference ground track (RGTs) is targeted "
            "in the polar regions once every 91 days.")
        # delta time of the crossing track
        IS2_atl11_tide[ptx][XT]['delta_time'] = delta_time['XT'].copy()
        IS2_atl11_fill[ptx][XT]['delta_time'] = delta_time['XT'].fill_value
        IS2_atl11_dims[ptx][XT]['delta_time'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT]['delta_time'] = {}
        IS2_atl11_tide_attrs[ptx][XT]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_tide_attrs[ptx][XT]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_tide_attrs[ptx][XT]['delta_time']['standard_name'] = "time"
        IS2_atl11_tide_attrs[ptx][XT]['delta_time']['calendar'] = "standard"
        IS2_atl11_tide_attrs[ptx][XT]['delta_time']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx][XT]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_tide_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt latitude longitude"
        # latitude of the crossover measurement
        IS2_atl11_tide[ptx][XT]['latitude'] = latitude['XT'].copy()
        IS2_atl11_fill[ptx][XT]['latitude'] = latitude['XT'].fill_value
        IS2_atl11_dims[ptx][XT]['latitude'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT]['latitude'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['units'] = "degrees_north"
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['long_name'] = "crossover latitude"
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['standard_name'] = "latitude"
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['valid_min'] = -90.0
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['valid_max'] = 90.0
        IS2_atl11_tide_attrs[ptx][XT]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude of the crossover measurement
        IS2_atl11_tide[ptx][XT]['longitude'] = longitude['XT'].copy()
        IS2_atl11_fill[ptx][XT]['longitude'] = longitude['XT'].fill_value
        IS2_atl11_dims[ptx][XT]['longitude'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT]['longitude'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['units'] = "degrees_east"
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['long_name'] = "crossover longitude"
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['standard_name'] = "longitude"
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['valid_min'] = -180.0
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['valid_max'] = 180.0
        IS2_atl11_tide_attrs[ptx][XT]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"
        # computed tide for the crossover measurement
        IS2_atl11_tide[ptx][XT][model.atl11] = tide['XT'].copy()
        IS2_atl11_fill[ptx][XT][model.atl11] = tide['XT'].fill_value
        IS2_atl11_dims[ptx][XT][model.atl11] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT][model.atl11] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT][model.atl11]['units'] = "meters"
        IS2_atl11_tide_attrs[ptx][XT][model.atl11]['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx][XT][model.atl11]['long_name'] = model.long_name
        IS2_atl11_tide_attrs[ptx][XT][model.atl11]['description'] = model.description
        IS2_atl11_tide_attrs[ptx][XT][model.atl11]['source'] = model.name
        IS2_atl11_tide_attrs[ptx][XT][model.atl11]['reference'] = model.reference
        IS2_atl11_tide_attrs[ptx][XT][model.atl11]['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # computed tide adjustments
        IS2_atl11_tide[ptx][XT]['tide_adj'] = tide_adj['XT'].copy()
        IS2_atl11_fill[ptx][XT]['tide_adj'] = tide_adj['XT'].fill_value
        IS2_atl11_dims[ptx][XT]['tide_adj'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['units'] = "1"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['long_name'] = "Ocean Tide Adjustment"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['description'] = ("Interpolated "
            "empirical adjustment applied to the ocean tides for ice flexure.")
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['source'] = tide_source
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['reference'] = tide_reference
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['coordinates'] = \
            "ref_pt delta_time latitude longitude"

    # print file information
    logger.info(f'\t{OUTPUT_FILE}')
    HDF5_ATL11_tide_write(IS2_atl11_tide, IS2_atl11_tide_attrs,
        CLOBBER=True, INPUT=os.path.basename(INPUT_FILE),
        FILL_VALUE=IS2_atl11_fill, DIMENSIONS=IS2_atl11_dims,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE))
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

# PURPOSE: outputting the tide values for ICESat-2 data to HDF5
def HDF5_ATL11_tide_write(IS2_atl11_tide, IS2_atl11_attrs, INPUT=None,
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
    for k,v in IS2_atl11_tide['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam pair
    pairs = [k for k in IS2_atl11_tide.keys() if bool(re.match(r'pt\d',k))]
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
            v = IS2_atl11_tide[ptx][k]
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

        # add to cycle_stats and crossing_track_data variables
        for key in ['cycle_stats','crossing_track_data']:
            fileID[ptx].create_group(key)
            h5[ptx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl11_attrs[ptx][key][att_name]
                fileID[ptx][key].attrs[att_name] = att_val
            for k,v in IS2_atl11_tide[ptx][key].items():
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
                        if (key == 'cycle_stats'):
                            h5[ptx][key][k].dims[i].attach_scale(h5[ptx][dim])
                        else:
                            h5[ptx][key][k].dims[i].attach_scale(h5[ptx][key][dim])
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
        lon = IS2_atl11_tide[ptx]['longitude']
        lat = IS2_atl11_tide[ptx]['latitude']
        delta_time = IS2_atl11_tide[ptx]['delta_time']
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
    # convert start and end time from ATLAS SDP seconds into GPS seconds
    atlas_sdp_gps_epoch=IS2_atl11_tide['ancillary_data']['atlas_sdp_gps_epoch']
    gps_seconds = atlas_sdp_gps_epoch + np.array([tmn,tmx])
    # calculate leap seconds
    leaps = pyTMD.time.count_leap_seconds(gps_seconds)
    # convert from seconds since 1980-01-06T00:00:00 to Julian days
    time_julian = 2400000.5 + pyTMD.time.convert_delta_time(gps_seconds - leaps,
        epoch1=(1980,1,6,0,0,0), epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    YY,MM,DD,HH,MN,SS = pyTMD.time.convert_julian(time_julian,format='tuple')
    # add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = f'{tmx-tmn:0.0f}'
    # add software information
    fileID.attrs['software_reference'] = pyTMD.version.project_name
    fileID.attrs['software_version'] = pyTMD.version.full_version
    fileID.attrs['software_revision'] = pyTMD.utilities.get_git_revision_hash()
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Applies interpolated tidal adjustment scale
            factors to ICESat-2 ATL11 annual land ice height data
            within ice sheet grounding zones
            """
    )
    # command line parameters
    group = parser.add_mutually_exclusive_group(required=True)
    # input ICESat-2 annual land ice height files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL11 file to run')
    # set adjustment file to use
    parser.add_argument('--flexure-file','-f',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Ice flexure file to use')
    # tide model to use
    group.add_argument('--tide','-T',
        metavar='TIDE', type=str,
        help='Tide model to use in correction')
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
        adjust_tides_ICESat2_ATL11(args.flexure_file, FILE,
            TIDE_MODEL=args.tide,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
