#!/usr/bin/env python
u"""
fit_tides_ICESat2_ATL11.py
Written by Tyler Sutterley (07/2022)
Fits tidal amplitudes to ICESat-2 data in ice sheet grounding zones

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -T X, --tide X: Tide model to use in correction
        CATS0201
        CATS2008
        CATS2022
        TPXO9-atlas
        TPXO9-atlas-v2
        TPXO9-atlas-v3
        TPXO9-atlas-v4
        TPXO9.1
        TPXO8-atlas
        TPXO7.2
        AODTM-5
        AOTIM-5
        AOTIM-5-2018
        GOT4.7
        GOT4.8
        GOT4.10
        FES2014
    -R X, --reanalysis X: Reanalysis model to run
        ERA-Interim: http://apps.ecmwf.int/datasets/data/interim-full-moda
        ERA5: http://apps.ecmwf.int/data-catalogues/era5/?class=ea
        MERRA-2: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
    -C, --crossovers: Run ATL11 Crossovers
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python (Spatial algorithms and data structures)
        https://docs.scipy.org/doc/
        https://docs.scipy.org/doc/scipy/reference/spatial.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/

PROGRAM DEPENDENCIES:
    read_ICESat2_ATL11.py: reads ICESat-2 annual land ice height data files
    time.py: utilities for calculating time operations

UPDATE HISTORY:
    Updated 07/2022: place some imports within try/except statements
    Updated 06/2022: include grounding zone adjusted DAC in HDF5 outputs
    Updated 05/2022: use argparse descriptions within documentation
    Updated 07/2021: add checks for data and fit quality
    Written 04/2021
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
import scipy.stats
import scipy.optimize

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

# PURPOSE: Find indices of common reference points between two lists
# Determines which across-track points correspond with the along-track
def common_reference_points(XT, AT):
    ind2 = [np.flatnonzero(XT == p) for p in AT]
    return ind2

# PURPOSE: read ICESat-2 annual land ice height data (ATL11) from NSIDC
# calculate mean elevation between all dates in file
# calculate inflexion point using elevation surface slopes
# use mean elevation to calculate elevation anomalies
# use anomalies to calculate inward and seaward limits of tidal flexure
def fit_tides_ICESat2(tide_dir, FILE,
    TIDE_MODEL=None,
    REANALYSIS=None,
    VERBOSE=False,
    MODE=0o775):

    # create logger
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # get tide model parameters
    model = pyTMD.model(tide_dir, verify=False).elevation(TIDE_MODEL)

    # print file information
    logging.info(os.path.basename(FILE))
    # read data from FILE
    mds1,attr1,pairs1 = is2tk.read_HDF5_ATL11(FILE, REFERENCE=True,
        CROSSOVERS=True, ATTRIBUTES=True, VERBOSE=VERBOSE)
    DIRECTORY = os.path.dirname(FILE)
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(FILE).pop()
    # file format for associated auxiliary files
    file_format = '{0}_{1}_{2}_{3}{4}_{5}{6}_{7}_{8}{9}.h5'

    # height threshold (filter points below 0m elevation)
    THRESHOLD = 0.0
    #  maximum height sigmas allowed in tidal adjustment fit
    sigma_tolerance = 0.5
    output_tolerance = 0.5

    # number of GPS seconds between the GPS epoch
    # and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = mds1['ancillary_data']['atlas_sdp_gps_epoch']

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
        IS2_atl11_tide['ancillary_data'][key] = mds1['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl11_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in attr1['ancillary_data'][key].items():
            IS2_atl11_tide_attrs['ancillary_data'][key][att_name] = att_val
    # HDF5 group name for across-track data
    XT = 'crossing_track_data'

    # for each input beam within the file
    for ptx in sorted(pairs1):
        # output data dictionaries for beam pair
        IS2_atl11_tide[ptx] = dict(cycle_stats=collections.OrderedDict(),
            crossing_track_data=collections.OrderedDict(),
            grounding_zone_data=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})
        IS2_atl11_dims[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})
        IS2_atl11_tide_attrs[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})

        # extract along-track and across-track variables
        ref_pt = {}
        latitude = {}
        longitude = {}
        delta_time = {}
        h_corr = {}
        h_sigma = {}
        quality_summary = {}
        tide_ocean = {}
        tide_adj = {}
        tide_adj_sigma = {}
        IB = {}
        groups = ['AT','XT']
        # number of average segments and number of included cycles
        # fill_value for invalid heights and corrections
        fv = attr1[ptx]['h_corr']['_FillValue']
        # shape of along-track data
        n_points,n_cycles = mds1[ptx]['delta_time'].shape
        # along-track (AT) reference point, latitude, longitude and time
        ref_pt['AT'] = mds1[ptx]['ref_pt'].copy()
        latitude['AT'] = np.ma.array(mds1[ptx]['latitude'],
            fill_value=attr1[ptx]['latitude']['_FillValue'])
        latitude['AT'].mask = (latitude['AT'] == latitude['AT'].fill_value)
        longitude['AT'] = np.ma.array(mds1[ptx]['longitude'],
            fill_value=attr1[ptx]['longitude']['_FillValue'])
        longitude['AT'].mask = (longitude['AT'] == longitude['AT'].fill_value)
        delta_time['AT'] = np.ma.array(mds1[ptx]['delta_time'],
            fill_value=attr1[ptx]['delta_time']['_FillValue'])
        delta_time['AT'].mask = (delta_time['AT'] == delta_time['AT'].fill_value)
        # corrected height and corrected height errors
        h_corr['AT'] = np.ma.array(mds1[ptx]['h_corr'],
            fill_value=attr1[ptx]['h_corr']['_FillValue'])
        h_corr['AT'].mask = (h_corr['AT'].data == h_corr['AT'].fill_value)
        h_sigma['AT'] = np.ma.array(mds1[ptx]['h_corr_sigma'],
            fill_value=attr1[ptx]['h_corr_sigma']['_FillValue'])
        h_sigma['AT'].mask = (h_sigma['AT'].data == h_sigma['AT'].fill_value)
        # quality summary
        quality_summary['AT'] = (mds1[ptx]['quality_summary'] == 0)
        # ocean corrections
        tide_ocean['AT'] = np.ma.array(mds1[ptx]['cycle_stats']['tide_ocean'],
            fill_value=attr1[ptx]['cycle_stats']['tide_ocean']['_FillValue'])
        tide_ocean['AT'].mask = (tide_ocean['AT'] == tide_ocean['AT'].fill_value)
        tide_adj['AT'] = np.ma.ones((n_points,n_cycles),
            fill_value=tide_ocean['AT'].fill_value)
        tide_adj['AT'].mask = (tide_ocean['AT'] == tide_ocean['AT'].fill_value)
        tide_adj_sigma['AT'] = np.ma.zeros((n_points,n_cycles),
            fill_value=tide_ocean['AT'].fill_value)
        tide_adj_sigma['AT'].mask = (tide_ocean['AT'] == tide_ocean['AT'].fill_value)
        # inverse barometer correction
        IB['AT'] = np.ma.array(mds1[ptx]['cycle_stats']['dac'],fill_value=0.0)
        IB['AT'].mask = (IB['AT'] == attr1[ptx]['cycle_stats']['dac']['_FillValue'])
        # ATL11 reference surface elevations (derived from ATL06)
        dem_h = mds1[ptx]['ref_surf']['dem_h']
        try:
            geoid_h = mds1[ptx]['ref_surf']['geoid_h']
        except ValueError:
            geoid_h = np.zeros_like(dem_h)

        # shape of across-track data
        n_cross, = mds1[ptx][XT]['delta_time'].shape
        # across-track (XT) reference point, latitude, longitude and time
        ref_pt['XT'] = mds1[ptx][XT]['ref_pt'].copy()
        latitude['XT'] = np.ma.array(mds1[ptx][XT]['latitude'],
            fill_value=attr1[ptx][XT]['latitude']['_FillValue'])
        latitude['XT'].mask = (latitude['XT'] == latitude['XT'].fill_value)
        longitude['XT'] = np.ma.array(mds1[ptx][XT]['longitude'],
            fill_value=attr1[ptx][XT]['longitude']['_FillValue'])
        latitude['XT'].mask = (latitude['XT'] == longitude['XT'].fill_value)
        delta_time['XT'] = np.ma.array(mds1[ptx][XT]['delta_time'],
            fill_value=attr1[ptx][XT]['delta_time']['_FillValue'])
        delta_time['XT'].mask = (delta_time['XT'] == delta_time['XT'].fill_value)
        # corrected height at crossovers
        h_corr['XT'] = np.ma.array(mds1[ptx][XT]['h_corr'],
            fill_value=attr1[ptx][XT]['h_corr']['_FillValue'])
        h_corr['XT'].mask = (h_corr['XT'].data == h_corr['XT'].fill_value)
        # across-track (XT) ocean corrections
        tide_ocean['XT'] = np.ma.array(mds1[ptx][XT]['tide_ocean'],
            fill_value=attr1[ptx][XT]['tide_ocean']['_FillValue'])
        tide_ocean['XT'].mask = (tide_ocean['XT'] == tide_ocean['XT'].fill_value)
        tide_adj['XT'] = np.ma.ones((n_cross),
            fill_value=tide_ocean['XT'].fill_value)
        tide_adj['XT'].mask = (tide_ocean['XT'] == tide_ocean['XT'].fill_value)
        tide_adj_sigma['XT'] = np.ma.zeros((n_cross),
            fill_value=tide_ocean['XT'].fill_value)
        tide_adj_sigma['XT'].mask = (tide_ocean['XT'] == tide_ocean['XT'].fill_value)
        # inverse barometer correction
        IB['XT'] = np.ma.array(mds1[ptx][XT]['dac'],fill_value=0.0)
        IB['XT'].mask = (IB['XT'] == attr1[ptx][XT]['dac']['_FillValue'])
        # find mapping between crossover and along-track reference points
        ref_indices = common_reference_points(ref_pt['XT'], ref_pt['AT'])

        # read buffered grounding zone mask
        a2 = (PRD,'GROUNDING_ZONE','MASK',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        f3 = os.path.join(DIRECTORY,file_format.format(*a2))
        # create data mask for grounding zone
        mds1[ptx]['subsetting'] = {}
        mds1[ptx]['subsetting'].setdefault('ice_gz',
            np.zeros((n_points),dtype=bool))
        # check that mask file exists
        try:
            mds2,attr2 = is2tk.read_HDF5_ATL11_pair(f3,ptx,
                ATTRIBUTES=True,VERBOSE=False,SUBSETTING=True)
        except:
            pass
        else:
            mds1[ptx]['subsetting']['ice_gz'] = \
                mds2[ptx]['subsetting']['ice_gz']
            B = attr2[ptx]['subsetting']['ice_gz']['source']

        # read tide model
        if TIDE_MODEL:
            # read tide model HDF5 file
            a3 = (PRD,TIDE_MODEL,'TIDES',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            f3 = os.path.join(DIRECTORY,file_format.format(*a3))
            # check that tide model file exists
            try:
                mds3,attr3 = is2tk.read_HDF5_ATL11_pair(f3,ptx,
                    VERBOSE=False,CROSSOVERS=True)
            except:
                # mask all values
                for group in groups:
                    tide_ocean[group].mask[:] = True
                pass
            else:
                tide_ocean['AT'].data[:] = mds3[ptx]['cycle_stats']['tide_ocean']
                tide_ocean['XT'].data[:] = mds3[ptx][XT]['tide_ocean']
            # source of tide model
            tide_source = TIDE_MODEL
            tide_reference = model.reference
        else:
            tide_source = 'ATL06'
            tide_reference = 'ATL06 ATBD'
        # set masks and fill values
        for group,val in tide_ocean.items():
            val.mask[:] = (val.data == val.fill_value)
            val.mask[:] |= (h_corr[group].data == h_corr[group].fill_value)
            val.data[val.mask] = val.fill_value

        # read inverse barometer correction
        if REANALYSIS:
            # read inverse barometer HDF5 file
            a4 = (PRD,REANALYSIS,'IB',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            f4 = os.path.join(DIRECTORY,file_format.format(*a4))
            # check that inverse barometer file exists
            try:
                mds4,attr4 = is2tk.read_HDF5_ATL11_pair(f4,ptx,
                    VERBOSE=False,CROSSOVERS=True)
            except:
                # mask all values
                for group in groups:
                    IB[group].mask[:] = True
                pass
            else:
                IB['AT'].data[:] = mds4[ptx]['cycle_stats']['ib']
                IB['XT'].data[:] = mds4[ptx][XT]['ib']
        # set masks and fill values
        for group,val in IB.items():
            val.mask[:] |= (val.data == val.fill_value)
            val.mask[:] |= (h_corr[group].data == h_corr[group].fill_value)
            val.data[val.mask] = val.fill_value

        # allocate for output tide times
        tide_time = {}
        # calculate tides for along-track and across-track data
        for track in ['AT','XT']:
            # convert time from ATLAS SDP to days relative to Jan 1, 1992
            gps_seconds = atlas_sdp_gps_epoch + delta_time[track]
            leap_seconds = is2tk.time.count_leap_seconds(gps_seconds)
            utc_seconds = gps_seconds - leap_seconds
            tide_time[track] = is2tk.time.convert_delta_time(utc_seconds,
                epoch1=(1980,1,6,0,0,0), epoch2=(1992,1,1,0,0,0), scale=1.0/86400.0)

        # for each ATL11 segment
        for s in range(n_points):
            # indices for crossover points
            i2 = np.squeeze(ref_indices[s])
            # create mask for valid points
            segment_mask = np.logical_not(h_corr['AT'].mask[s,:])
            # segment_mask &= np.logical_not(IB['AT'].mask[s,:])
            segment_mask &= np.logical_not(tide_ocean['AT'].mask[s,:])
            segment_mask &= quality_summary['AT'][s,:]
            segment_mask &= (h_corr['AT'].data[s,:] > THRESHOLD)
            segment_mask &= (h_sigma['AT'].data[s,:] < sigma_tolerance)
            segment_mask &= mds1[ptx]['subsetting']['ice_gz'][s]
            if not np.any(segment_mask):
                # set masks
                tide_adj['AT'].mask[s,:] = True
                tide_adj_sigma['AT'].mask[s,:] = True
                if np.any(i2):
                    tide_adj['XT'].mask[i2] = True
                    tide_adj_sigma['XT'].mask[i2] = True
                # continue to next iteration
                continue
            # indices for valid points within segment
            i1, = np.nonzero(segment_mask)
            # height referenced to geoid
            h1 = h_corr['AT'].data[s,i1] - geoid_h[s]
            h2 = np.atleast_1d(h_corr['XT'].data[i2]) - geoid_h[s]
            n1 = len(h1)
            n2 = len(h2)
            # tide time
            t1 = tide_time['AT'].data[s,i1]
            t2 = np.atleast_1d(tide_time['XT'].data[i2])
            # combined tide and dac height
            ot1 = tide_ocean['AT'].data[s,i1] + IB['AT'].data[s,i1]
            ot2 = np.atleast_1d(tide_ocean['XT'].data[i2] + IB['XT'].data[i2])

            # combine along-track and across-track variables
            if np.any(i2):
                h = np.concatenate((h1,h2),axis=0)
                t = np.concatenate((t1,t2),axis=0)
                tide = np.concatenate((ot1,ot2),axis=0)
            else:
                h = np.copy(h1)
                t = np.copy(t1)
                tide = np.copy(ot1)

            # use linear least-squares with bounds on the variables
            # create design matrix
            p0 = np.ones_like(t)
            p1 = 365.25*(t - np.median(t))
            DMAT = np.c_[tide,p0,p1]
            # check if there are enough unique dates for fit
            u_days = np.unique(np.round(p1)/365.25)
            if (len(u_days) <= 3):
                # set masks
                tide_adj['AT'].mask[s,:] = True
                tide_adj_sigma['AT'].mask[s,:] = True
                if np.any(i2):
                    tide_adj['XT'].mask[i2] = True
                    tide_adj_sigma['XT'].mask[i2] = True
                # continue to next iteration
                continue
            n_max,n_terms = np.shape(DMAT)
            # nu = Degrees of Freedom
            nu = n_max - n_terms

            # tuple for parameter bounds (lower and upper)
            lb,ub = ([0.0,np.min(h),-2.0],[1.0,np.max(h),2.0])
            # use linear least-squares with bounds on the variables
            try:
                results = scipy.optimize.lsq_linear(DMAT, h, bounds=(lb,ub))
                # Mean square error
                MSE = np.sum(results['fun']**2)/np.float64(nu)
                # Covariance Matrix
                # Multiplying the design matrix by itself
                Hinv = np.linalg.inv(np.dot(np.transpose(DMAT),DMAT))
            except:
                # set masks
                tide_adj['AT'].mask[s,:] = True
                tide_adj_sigma['AT'].mask[s,:] = True
                if np.any(i2):
                    tide_adj['XT'].mask[i2] = True
                    tide_adj_sigma['XT'].mask[i2] = True
                # continue to next iteration
                continue
            else:
                # cadj,sadj,H,dH = np.copy(results['x'])
                adj,H,dH = np.copy(results['x'])
                # calculate mean square error
                MSE = np.sum(results['fun']**2)/nu
                # standard error from covariance matrix
                adj_sigma,*_ = np.sqrt(MSE*np.diag(Hinv))
            # check that errors are smaller than tolerance
            if (adj_sigma > output_tolerance):
                continue
            # extract along-track and across-track tide
            tide_ocean['AT'].data[s,i1] *= adj
            IB['AT'].data[s,i1] *= adj
            tide_adj['AT'][s,i1] = np.copy(adj)
            tide_adj_sigma['AT'][s,i1] = np.copy(adj_sigma)
            if np.any(i2):
                tide_ocean['XT'].data[i2] *= adj
                IB['XT'].data[i2] *= adj
                tide_adj['XT'][i2] = np.copy(adj)
                tide_adj_sigma['XT'][i2] = np.copy(adj_sigma)

        # group attributes for beam
        IS2_atl11_tide_attrs[ptx]['description'] = ('Contains the primary science parameters '
            'for this data set')
        IS2_atl11_tide_attrs[ptx]['beam_pair'] = attr1[ptx]['beam_pair']
        IS2_atl11_tide_attrs[ptx]['ReferenceGroundTrack'] = attr1[ptx]['ReferenceGroundTrack']
        IS2_atl11_tide_attrs[ptx]['first_cycle'] = attr1[ptx]['first_cycle']
        IS2_atl11_tide_attrs[ptx]['last_cycle'] = attr1[ptx]['last_cycle']
        IS2_atl11_tide_attrs[ptx]['equatorial_radius'] = attr1[ptx]['equatorial_radius']
        IS2_atl11_tide_attrs[ptx]['polar_radius'] = attr1[ptx]['polar_radius']

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
        IS2_atl11_tide[ptx]['cycle_number'] = mds1[ptx]['cycle_number'].copy()
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
        # computed tide with fit
        IS2_atl11_tide[ptx]['cycle_stats']['tide_ocean'] = tide_ocean['AT'].copy()
        IS2_atl11_fill[ptx]['cycle_stats']['tide_ocean'] = tide_ocean['AT'].fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['tide_ocean'] = ['ref_pt','cycle_number']
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_ocean'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_ocean']['units'] = "meters"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_ocean']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_ocean']['long_name'] = "Ocean Tide"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_ocean']['description'] = ("Ocean Tides with "
            "Near-Grounding Zone fit that includes diurnal and semi-diurnal (harmonic analysis), "
            "and longer period tides (dynamic and self-consistent equilibrium).")
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_ocean']['source'] = tide_source
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_ocean']['reference'] = tide_reference
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_ocean']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"
        # computed dac with fit
        IS2_atl11_tide[ptx]['cycle_stats']['dac'] = IB['AT'].copy()
        IS2_atl11_fill[ptx]['cycle_stats']['dac'] = IB['AT'].fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['dac'] = ['ref_pt','cycle_number']
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['dac'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['dac']['units'] = "meters"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['dac']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['dac']['long_name'] = "Dynamic atmosphere correction "
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['dac']['description'] = ("Weighted-average "
            "dynamic atmosphere correction for each pass with Near-Grounding Zone fit.")
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['dac']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['dac']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"
        # computed tide adjustments
        IS2_atl11_tide[ptx]['cycle_stats']['tide_adj'] = tide_adj['AT'].copy()
        IS2_atl11_fill[ptx]['cycle_stats']['tide_adj'] = tide_adj['AT'].fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['tide_adj'] = ['ref_pt','cycle_number']
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['units'] = "1"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['long_name'] = "Ocean Tide Adjustment"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['description'] = ("Empirical "
            "adjustment applied to the ocean tides for Near-Grounding Zone data.")
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['source'] = tide_source
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['reference'] = tide_reference
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"
        # computed tide adjustment uncertainty
        IS2_atl11_tide[ptx]['cycle_stats']['tide_adj_sigma'] = tide_adj_sigma['AT'].copy()
        IS2_atl11_fill[ptx]['cycle_stats']['tide_adj_sigma'] = tide_adj_sigma['AT'].fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['tide_adj_sigma'] = ['ref_pt','cycle_number']
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj_sigma'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj_sigma']['units'] = "1"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj_sigma']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj_sigma']['long_name'] = \
            "Ocean Tide Adjustment Uncertainty"
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj_sigma']['description'] = ("Empirical "
            "adjustment uncertainty from the bounded least-squares fit.")
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj_sigma']['source'] = tide_source
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj_sigma']['reference'] = tide_reference
        IS2_atl11_tide_attrs[ptx]['cycle_stats']['tide_adj_sigma']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"

        # crossing track variables
        IS2_atl11_tide_attrs[ptx][XT]['Description'] = ("The crossing_track_data "
            "subgroup contains elevation data at crossover locations. These are "
            "locations where two ICESat-2 pair tracks cross, so data are available "
            "from both the datum track, for which the granule was generated, and "
            "from the crossing track.")
        IS2_atl11_tide_attrs[ptx][XT]['data_rate'] = ("Data within this group are "
            "stored at the average segment rate.")

        # reference point
        IS2_atl11_tide[ptx][XT]['ref_pt'] = mds1[ptx][XT]['ref_pt'].copy()
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
        IS2_atl11_tide[ptx][XT]['rgt'] = mds1[ptx][XT]['rgt'].copy()
        IS2_atl11_fill[ptx][XT]['rgt'] = attr1[ptx][XT]['rgt']['_FillValue']
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
        IS2_atl11_tide[ptx][XT]['cycle_number'] = mds1[ptx][XT]['cycle_number'].copy()
        IS2_atl11_fill[ptx][XT]['cycle_number'] = attr1[ptx][XT]['cycle_number']['_FillValue']
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
        IS2_atl11_tide_attrs[ptx][XT]['delta_time']['coordinates'] = \
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
        # computed tide from fit for the crossover measurement
        IS2_atl11_tide[ptx][XT]['tide_ocean'] = tide_ocean['XT'].copy()
        IS2_atl11_fill[ptx][XT]['tide_ocean'] = tide_ocean['XT'].fill_value
        IS2_atl11_dims[ptx][XT]['tide_ocean'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT]['tide_ocean'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['tide_ocean']['units'] = "meters"
        IS2_atl11_tide_attrs[ptx][XT]['tide_ocean']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx][XT]['tide_ocean']['long_name'] = "Ocean Tide"
        IS2_atl11_tide_attrs[ptx][XT]['tide_ocean']['description'] = ("Ocean Tides with "
            "Near-Grounding Zone fit that includes diurnal and semi-diurnal (harmonic analysis), "
            "and longer period tides (dynamic and self-consistent equilibrium).")
        IS2_atl11_tide_attrs[ptx][XT]['tide_ocean']['source'] = tide_source
        IS2_atl11_tide_attrs[ptx][XT]['tide_ocean']['reference'] = tide_reference
        IS2_atl11_tide_attrs[ptx][XT]['tide_ocean']['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # computed dac with fit for the crossover measurement
        IS2_atl11_tide[ptx][XT]['dac'] = IB['XT'].copy()
        IS2_atl11_fill[ptx][XT]['dac'] = IB['XT'].fill_value
        IS2_atl11_dims[ptx][XT]['dac'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT]['dac'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['dac']['units'] = "meters"
        IS2_atl11_tide_attrs[ptx][XT]['dac']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx][XT]['dac']['long_name'] = "Dynamic atmosphere correction"
        IS2_atl11_tide_attrs[ptx][XT]['dac']['description'] = ("Weighted-average "
            "dynamic atmosphere correction for each pass with Near-Grounding Zone fit.")
        IS2_atl11_tide_attrs[ptx][XT]['dac']['source'] = "ATL06"
        IS2_atl11_tide_attrs[ptx][XT]['dac']['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # computed tide adjustments
        IS2_atl11_tide[ptx][XT]['tide_adj'] = tide_adj['XT'].copy()
        IS2_atl11_fill[ptx][XT]['tide_adj'] = tide_adj['XT'].fill_value
        IS2_atl11_dims[ptx][XT]['tide_adj'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['units'] = "1"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['long_name'] = "Ocean Tide Adjustment"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['description'] = ("Empirical "
            "adjustment applied to the ocean tides for Near-Grounding Zone data.")
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['source'] = tide_source
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['reference'] = tide_reference
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj']['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # computed tide adjustment uncertainty
        IS2_atl11_tide[ptx][XT]['tide_adj_sigma'] = tide_adj_sigma['XT'].copy()
        IS2_atl11_fill[ptx][XT]['tide_adj_sigma'] = tide_adj_sigma['XT'].fill_value
        IS2_atl11_dims[ptx][XT]['tide_adj_sigma'] = ['ref_pt']
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj_sigma'] = collections.OrderedDict()
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj_sigma']['units'] = "1"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj_sigma']['contentType'] = "referenceInformation"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj_sigma']['long_name'] = \
            "Ocean Tide Adjustment Uncertainty"
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj_sigma']['description'] = ("Empirical "
            "adjustment uncertainty from the bounded least-squares fit.")
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj_sigma']['source'] = tide_source
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj_sigma']['reference'] = tide_reference
        IS2_atl11_tide_attrs[ptx][XT]['tide_adj_sigma']['coordinates'] = \
            "ref_pt delta_time latitude longitude"

    # output flexure correction HDF5 file
    args = (PRD,TIDE_MODEL,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
    file_format = '{0}_{1}_FIT_TIDES_{2}{3}_{4}{5}_{6}_{7}{8}.h5'
    OUTPUT_FILE = os.path.join(DIRECTORY,file_format.format(*args))
    # print file information
    logging.info(f'\t{OUTPUT_FILE}')
    HDF5_ATL11_corr_write(IS2_atl11_tide, IS2_atl11_tide_attrs,
        CLOBBER=True, INPUT=os.path.basename(FILE),
        CROSSOVERS=True, FILL_VALUE=IS2_atl11_fill, DIMENSIONS=IS2_atl11_dims,
        FILENAME=OUTPUT_FILE)
    # change the permissions mode
    os.chmod(OUTPUT_FILE, MODE)

# PURPOSE: outputting the correction values for ICESat-2 data to HDF5
def HDF5_ATL11_corr_write(IS2_atl11_corr, IS2_atl11_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, GROUNDING_ZONE=False,
    CROSSOVERS=False, CLOBBER=False):
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
    for k,v in IS2_atl11_corr['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam pair
    pairs = [k for k in IS2_atl11_corr.keys() if bool(re.match(r'pt\d',k))]
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
            v = IS2_atl11_corr[ptx][k]
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

        # add to cycle_stats variables
        groups = ['cycle_stats']
        # if there were valid fits: add to grounding_zone_data variables
        if GROUNDING_ZONE:
            groups.append('grounding_zone_data')
        # if running crossovers: add to crossing_track_data variables
        if CROSSOVERS:
            groups.append('crossing_track_data')
        for key in groups:
            fileID[ptx].create_group(key)
            h5[ptx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl11_attrs[ptx][key][att_name]
                fileID[ptx][key].attrs[att_name] = att_val
            for k,v in IS2_atl11_corr[ptx][key].items():
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
        lon = IS2_atl11_corr[ptx]['longitude']
        lat = IS2_atl11_corr[ptx]['latitude']
        delta_time = IS2_atl11_corr[ptx]['delta_time']
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
    atlas_sdp_gps_epoch=IS2_atl11_corr['ancillary_data']['atlas_sdp_gps_epoch']
    gps_seconds = atlas_sdp_gps_epoch + np.array([tmn,tmx])
    # calculate leap seconds
    leaps = is2tk.time.count_leap_seconds(gps_seconds)
    # convert from seconds since 1980-01-06T00:00:00 to Julian days
    MJD = is2tk.time.convert_delta_time(gps_seconds - leaps,
        epoch1=(1980,1,6,0,0,0), epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    YY,MM,DD,HH,MN,SS = is2tk.time.convert_julian(MJD + 2400000.5,
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
    fileID.attrs['software_reference'] = pyTMD.version.project_name
    fileID.attrs['software_version'] = pyTMD.version.full_version
    fileID.attrs['software_revision'] = pyTMD.utilities.get_git_revision_hash()
    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create arguments parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Fits tidal amplitudes to ICESat-2 data in
            ice sheet grounding zones
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL11 file to run')
    # directory with tide data
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # tide model to use
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, default='CATS2008',
        help='Tide model to use in correction')
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str,
        help='Reanalysis model to use in inverse-barometer correction')
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
        fit_tides_ICESat2(args.directory, FILE,
            TIDE_MODEL=args.tide,
            REANALYSIS=args.reanalysis,
            VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
