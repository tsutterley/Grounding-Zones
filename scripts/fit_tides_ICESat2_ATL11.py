#!/usr/bin/env python
u"""
fit_tides_ICESat2_ATL11.py
Written by Tyler Sutterley (04/2021)
Fits tidal amplitudes to ICESat-2 data in ice sheet grounding zones

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -T X, --tide X: Tide model to use in correction
        CATS0201
        CATS2008
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
    matplotlib: Python 2D plotting library
        http://matplotlib.org/
        https://github.com/matplotlib/matplotlib
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/

PROGRAM DEPENDENCIES:
    read_ICESat2_ATL11.py: reads ICESat-2 annual land ice height data files
    time.py: utilities for calculating time operations
    utilities: download and management utilities for syncing files

UPDATE HISTORY:
    Written 04/2021
"""
from __future__ import print_function

import sys
import os
import re
import h5py
import datetime
import argparse
import numpy as np
import collections
import scipy.stats
import scipy.optimize
import icesat2_toolkit.time
import pyTMD.utilities
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.read_tide_model import extract_tidal_constants
from pyTMD.read_netcdf_model import extract_netcdf_constants
from pyTMD.read_GOT_model import extract_GOT_constants
from pyTMD.read_FES_model import extract_FES_constants
from pyTMD.load_constituent import load_constituent
from pyTMD.load_nodal_corrections import load_nodal_corrections
from pyTMD.infer_minor_corrections import infer_minor_corrections
from icesat2_toolkit.read_ICESat2_ATL11 import read_HDF5_ATL11, \
    read_HDF5_ATL11_pair

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
def fit_tides_ICESat2(tide_dir, FILE, TIDE_MODEL=None, REANALYSIS=None,
    METHOD='spline', VERBOSE=False, MODE=0o775):
    # print file information
    print(os.path.basename(FILE)) if VERBOSE else None
    # read data from FILE
    mds1,attr1,pairs1 = read_HDF5_ATL11(FILE, REFERENCE=True,
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

    # select between tide models
    if (TIDE_MODEL == 'CATS0201'):
        grid_file = os.path.join(tide_dir,'cats0201_tmd','grid_CATS')
        model_file = os.path.join(tide_dir,'cats0201_tmd','h0_CATS02_01')
        tide_reference = 'https://mail.esr.org/polar_tide_models/Model_CATS0201.html'
        model_format = 'OTIS'
        EPSG = '4326'
        TYPE = 'z'
    elif (TIDE_MODEL == 'CATS2008'):
        grid_file = os.path.join(tide_dir,'CATS2008','grid_CATS2008')
        model_file = os.path.join(tide_dir,'CATS2008','hf.CATS2008.out')
        tide_reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/cats2008/')
        model_format = 'OTIS'
        EPSG = 'CATS2008'
        TYPE = 'z'
    elif (TIDE_MODEL == 'TPXO9-atlas'):
        model_directory = os.path.join(tide_dir,'TPXO9_atlas')
        grid_file = os.path.join(model_directory,'grid_tpxo9_atlas.nc.gz')
        model_files = ['h_q1_tpxo9_atlas_30.nc.gz','h_o1_tpxo9_atlas_30.nc.gz',
            'h_p1_tpxo9_atlas_30.nc.gz','h_k1_tpxo9_atlas_30.nc.gz',
            'h_n2_tpxo9_atlas_30.nc.gz','h_m2_tpxo9_atlas_30.nc.gz',
            'h_s2_tpxo9_atlas_30.nc.gz','h_k2_tpxo9_atlas_30.nc.gz',
            'h_m4_tpxo9_atlas_30.nc.gz','h_ms4_tpxo9_atlas_30.nc.gz',
            'h_mn4_tpxo9_atlas_30.nc.gz','h_2n2_tpxo9_atlas_30.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        tide_reference = 'http://volkov.oce.orst.edu/tides/tpxo9_atlas.html'
        model_format = 'netcdf'
        TYPE = 'z'
        SCALE = 1.0/1000.0
        GZIP = True
    elif (TIDE_MODEL == 'TPXO9-atlas-v2'):
        model_directory = os.path.join(tide_dir,'TPXO9_atlas_v2')
        grid_file = os.path.join(model_directory,'grid_tpxo9_atlas_30_v2.nc.gz')
        model_files = ['h_q1_tpxo9_atlas_30_v2.nc.gz','h_o1_tpxo9_atlas_30_v2.nc.gz',
            'h_p1_tpxo9_atlas_30_v2.nc.gz','h_k1_tpxo9_atlas_30_v2.nc.gz',
            'h_n2_tpxo9_atlas_30_v2.nc.gz','h_m2_tpxo9_atlas_30_v2.nc.gz',
            'h_s2_tpxo9_atlas_30_v2.nc.gz','h_k2_tpxo9_atlas_30_v2.nc.gz',
            'h_m4_tpxo9_atlas_30_v2.nc.gz','h_ms4_tpxo9_atlas_30_v2.nc.gz',
            'h_mn4_tpxo9_atlas_30_v2.nc.gz','h_2n2_tpxo9_atlas_30_v2.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        tide_reference = 'https://www.tpxo.net/global/tpxo9-atlas'
        model_format = 'netcdf'
        TYPE = 'z'
        SCALE = 1.0/1000.0
        GZIP = True
    elif (TIDE_MODEL == 'TPXO9-atlas-v3'):
        model_directory = os.path.join(tide_dir,'TPXO9_atlas_v3')
        grid_file = os.path.join(model_directory,'grid_tpxo9_atlas_30_v3.nc.gz')
        model_files = ['h_q1_tpxo9_atlas_30_v3.nc.gz','h_o1_tpxo9_atlas_30_v3.nc.gz',
            'h_p1_tpxo9_atlas_30_v3.nc.gz','h_k1_tpxo9_atlas_30_v3.nc.gz',
            'h_n2_tpxo9_atlas_30_v3.nc.gz','h_m2_tpxo9_atlas_30_v3.nc.gz',
            'h_s2_tpxo9_atlas_30_v3.nc.gz','h_k2_tpxo9_atlas_30_v3.nc.gz',
            'h_m4_tpxo9_atlas_30_v3.nc.gz','h_ms4_tpxo9_atlas_30_v3.nc.gz',
            'h_mn4_tpxo9_atlas_30_v3.nc.gz','h_2n2_tpxo9_atlas_30_v3.nc.gz',
            'h_mf_tpxo9_atlas_30_v3.nc.gz','h_mm_tpxo9_atlas_30_v3.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        tide_reference = 'https://www.tpxo.net/global/tpxo9-atlas'
        model_format = 'netcdf'
        TYPE = 'z'
        SCALE = 1.0/1000.0
        GZIP = True
    elif (TIDE_MODEL == 'TPXO9-atlas-v4'):
        model_directory = os.path.join(tide_dir,'TPXO9_atlas_v4')
        grid_file = os.path.join(model_directory,'grid_tpxo9_atlas_30_v4')
        model_files = ['h_q1_tpxo9_atlas_30_v4','h_o1_tpxo9_atlas_30_v4',
            'h_p1_tpxo9_atlas_30_v4','h_k1_tpxo9_atlas_30_v4',
            'h_n2_tpxo9_atlas_30_v4','h_m2_tpxo9_atlas_30_v4',
            'h_s2_tpxo9_atlas_30_v4','h_k2_tpxo9_atlas_30_v4',
            'h_m4_tpxo9_atlas_30_v4','h_ms4_tpxo9_atlas_30_v4',
            'h_mn4_tpxo9_atlas_30_v4','h_2n2_tpxo9_atlas_30_v4',
            'h_mf_tpxo9_atlas_30_v4','h_mm_tpxo9_atlas_30_v4']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        tide_reference = 'https://www.tpxo.net/global/tpxo9-atlas'
        model_format = 'OTIS'
        EPSG = '4326'
        TYPE = 'z'
    elif (TIDE_MODEL == 'TPXO9.1'):
        grid_file = os.path.join(tide_dir,'TPXO9.1','DATA','grid_tpxo9')
        model_file = os.path.join(tide_dir,'TPXO9.1','DATA','h_tpxo9.v1')
        tide_reference = 'http://volkov.oce.orst.edu/tides/global.html'
        model_format = 'OTIS'
        EPSG = '4326'
        TYPE = 'z'
    elif (TIDE_MODEL == 'TPXO8-atlas'):
        grid_file = os.path.join(tide_dir,'tpxo8_atlas','grid_tpxo8atlas_30_v1')
        model_file = os.path.join(tide_dir,'tpxo8_atlas','hf.tpxo8_atlas_30_v1')
        tide_reference = 'http://volkov.oce.orst.edu/tides/tpxo8_atlas.html'
        model_format = 'ATLAS'
        EPSG = '4326'
        TYPE = 'z'
    elif (TIDE_MODEL == 'TPXO7.2'):
        grid_file = os.path.join(tide_dir,'TPXO7.2_tmd','grid_tpxo7.2')
        model_file = os.path.join(tide_dir,'TPXO7.2_tmd','h_tpxo7.2')
        tide_reference = 'http://volkov.oce.orst.edu/tides/global.html'
        model_format = 'OTIS'
        EPSG = '4326'
        TYPE = 'z'
    elif (TIDE_MODEL == 'AODTM-5'):
        grid_file = os.path.join(tide_dir,'aodtm5_tmd','grid_Arc5km')
        model_file = os.path.join(tide_dir,'aodtm5_tmd','h0_Arc5km.oce')
        tide_reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/aodtm-5/')
        model_format = 'OTIS'
        EPSG = 'PSNorth'
        TYPE = 'z'
    elif (TIDE_MODEL == 'AOTIM-5'):
        grid_file = os.path.join(tide_dir,'aotim5_tmd','grid_Arc5km')
        model_file = os.path.join(tide_dir,'aotim5_tmd','h_Arc5km.oce')
        tide_reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/aotim-5/')
        model_format = 'OTIS'
        EPSG = 'PSNorth'
        TYPE = 'z'
    elif (TIDE_MODEL == 'AOTIM-5-2018'):
        grid_file = os.path.join(tide_dir,'Arc5km2018','grid_Arc5km2018')
        model_file = os.path.join(tide_dir,'Arc5km2018','h_Arc5km2018')
        tide_reference = ('https://www.esr.org/research/polar-tide-models/'
            'list-of-polar-tide-models/aotim-5/')
        model_format = 'OTIS'
        EPSG = 'PSNorth'
        TYPE = 'z'
    elif (TIDE_MODEL == 'GOT4.7'):
        model_directory = os.path.join(tide_dir,'GOT4.7','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        tide_reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        model_format = 'GOT'
        SCALE = 1.0/100.0
        GZIP = True
    elif (TIDE_MODEL == 'GOT4.8'):
        model_directory = os.path.join(tide_dir,'got4.8','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        tide_reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        model_format = 'GOT'
        SCALE = 1.0/100.0
        GZIP = True
    elif (TIDE_MODEL == 'GOT4.10'):
        model_directory = os.path.join(tide_dir,'GOT4.10c','grids_oceantide')
        model_files = ['q1.d.gz','o1.d.gz','p1.d.gz','k1.d.gz','n2.d.gz',
            'm2.d.gz','s2.d.gz','k2.d.gz','s1.d.gz','m4.d.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        tide_reference = ('https://denali.gsfc.nasa.gov/personal_pages/ray/'
            'MiscPubs/19990089548_1999150788.pdf')
        model_format = 'GOT'
        SCALE = 1.0/100.0
        GZIP = True
    elif (TIDE_MODEL == 'FES2014'):
        model_directory = os.path.join(tide_dir,'fes2014','ocean_tide')
        model_files = ['2n2.nc.gz','eps2.nc.gz','j1.nc.gz','k1.nc.gz',
            'k2.nc.gz','l2.nc.gz','la2.nc.gz','m2.nc.gz','m3.nc.gz','m4.nc.gz',
            'm6.nc.gz','m8.nc.gz','mf.nc.gz','mks2.nc.gz','mm.nc.gz',
            'mn4.nc.gz','ms4.nc.gz','msf.nc.gz','msqm.nc.gz','mtm.nc.gz',
            'mu2.nc.gz','n2.nc.gz','n4.nc.gz','nu2.nc.gz','o1.nc.gz','p1.nc.gz',
            'q1.nc.gz','r2.nc.gz','s1.nc.gz','s2.nc.gz','s4.nc.gz','sa.nc.gz',
            'ssa.nc.gz','t2.nc.gz']
        model_file = [os.path.join(model_directory,m) for m in model_files]
        c = ['2n2','eps2','j1','k1','k2','l2','lambda2','m2','m3','m4','m6',
            'm8','mf','mks2','mm','mn4','ms4','msf','msqm','mtm','mu2','n2',
            'n4','nu2','o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
        tide_reference = ('https://www.aviso.altimetry.fr/en/data/products'
            'auxiliary-products/global-tide-fes.html')
        model_format = 'FES'
        TYPE = 'z'
        SCALE = 1.0/100.0
        GZIP = True

    # number of GPS seconds between the GPS epoch
    # and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = mds1['ancillary_data']['atlas_sdp_gps_epoch']
    # delta time (TT - UT1) file
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])

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
        quality_summary = {}
        tide_ocean = {}
        tide_error = {}
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
        # corrected height
        h_corr['AT'] = np.ma.array(mds1[ptx]['h_corr'],
            fill_value=attr1[ptx]['h_corr']['_FillValue'])
        h_corr['AT'].mask = (h_corr['AT'].data == h_corr['AT'].fill_value)
        # quality summary
        quality_summary['AT'] = (mds1[ptx]['quality_summary'] == 0)
        # ocean corrections
        tide_ocean['AT'] = np.ma.array(mds1[ptx]['cycle_stats']['tide_ocean'],
            fill_value=attr1[ptx]['cycle_stats']['tide_ocean']['_FillValue'])
        tide_ocean['AT'].mask = (tide_ocean['AT'] == tide_ocean['AT'].fill_value)
        tide_ocean['AT'] = np.ma.zeros((n_points,n_cycles),
            fill_value=tide_ocean['AT'].fill_value)
        tide_ocean['AT'].mask = (tide_ocean['AT'] == tide_ocean['AT'].fill_value)
        IB['AT'] = np.ma.array(mds1[ptx]['cycle_stats']['dac'],fill_value=0.0)
        IB['AT'].mask = (IB['AT'] == attr1[ptx]['cycle_stats']['dac']['_FillValue'])
        # ATL11 reference surface elevations (derived from ATL06)
        dem_h = mds1[ptx]['ref_surf']['dem_h']
        # geoid_h = mds1[ptx]['ref_surf']['geoid_h']

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
        tide_error['XT'] = np.ma.zeros((n_cross),
            fill_value=tide_ocean['XT'].fill_value)
        tide_error['XT'].mask = (tide_ocean['XT'] == tide_ocean['XT'].fill_value)
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
            mds2,attr2 = read_HDF5_ATL11_pair(f3,ptx,
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
            # check that sea level file exists
            try:
                mds3,attr3 = read_HDF5_ATL11_pair(f3,ptx,
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
            # check that sea level file exists
            try:
                mds4,attr4 = read_HDF5_ATL11_pair(f4,ptx,
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

        # allocate for output tidal amplitude and phase
        amp,ph = ({},{})
        # allocate for output tide times
        tide_time,deltat = ({},{})
        # calculate tides for along-track and across-track data
        for track in ['AT','XT']:
            # convert time from ATLAS SDP to days relative to Jan 1, 1992
            gps_seconds = atlas_sdp_gps_epoch + delta_time[track]
            leap_seconds = icesat2_toolkit.time.count_leap_seconds(gps_seconds)
            utc_seconds = gps_seconds - leap_seconds
            tide_time[track] = icesat2_toolkit.time.convert_delta_time(utc_seconds,
                epoch1=(1980,1,6,0,0,0), epoch2=(1992,1,1,0,0,0), scale=1.0/86400.0)
        #     # read tidal constants and interpolate to grid points
        #     if model_format in ('OTIS','ATLAS'):
        #         amp[track],ph[track],D,c = extract_tidal_constants(longitude[track],
        #             latitude[track], grid_file, model_file, EPSG, TYPE=TYPE,
        #             METHOD=METHOD, EXTRAPOLATE=True, GRID=model_format)
        #         deltat[track] = np.zeros_like(tide_time[track])
        #     elif (model_format == 'netcdf'):
        #         amp[track],ph[track],D,c = extract_netcdf_constants(longitude[track],
        #             latitude[track], grid_file, model_file, TYPE=TYPE, METHOD=METHOD,
        #             EXTRAPOLATE=True, SCALE=SCALE, GZIP=GZIP)
        #         deltat[track] = np.zeros_like(tide_time[track])
        #     elif (model_format == 'GOT'):
        #         amp[track],ph[track],c = extract_GOT_constants(longitude[track],
        #             latitude[track], model_file, METHOD=METHOD, EXTRAPOLATE=True,
        #             SCALE=SCALE, GZIP=GZIP)
        #         # interpolate delta times from calendar dates to tide time
        #         deltat[track] = calc_delta_time(delta_file, tide_time)
        #     elif (model_format == 'FES'):
        #         amp[track],ph[track] = extract_FES_constants(longitude[track],
        #             latitude[track], model_file, TYPE=TYPE, VERSION=TIDE_MODEL,
        #             METHOD=METHOD, EXTRAPOLATE=True, SCALE=SCALE, GZIP=GZIP)
        #         # interpolate delta times from calendar dates to tide time
        #         deltat[track] = calc_delta_time(delta_file, tide_time)

        # # number of constituents
        # nc = len(c)
        # for each ATL11 segment
        for s in range(n_points):
            # create mask for valid points
            segment_mask = np.logical_not(h_corr['AT'].mask[s,:])
            # segment_mask &= np.logical_not(IB['AT'].mask[s,:])
            segment_mask &= np.logical_not(tide_ocean['AT'].mask[s,:])
            segment_mask &= quality_summary['AT'][s,:]
            segment_mask &= (h_corr['AT'].data[s,:] > THRESHOLD)
            segment_mask &= mds1[ptx]['subsetting']['ice_gz'][s]
            if not np.any(segment_mask):
                continue
            i1, = np.nonzero(segment_mask)
            i2 = np.squeeze(ref_indices[s])
            # height referenced to geoid
            h1 = h_corr['AT'].data[s,i1] - IB['AT'].data[s,i1] #- geoid_h[s]
            h2 = np.atleast_1d(h_corr['XT'].data[i2] - IB['XT'].data[i2]) #- geoid_h[s]
            n1 = len(h1)
            n2 = len(h2)
            # tide time
            t1 = tide_time['AT'].data[s,i1]
            t2 = np.atleast_1d(tide_time['XT'].data[i2])
            # # tide delta time (TT - UT1)
            # dt1 = deltat['AT'][s,i1]
            # dt2 = np.atleast_1d(deltat['XT'][i2])
            # tide height
            ot1 = tide_ocean['AT'].data[s,i1]
            ot2 = np.atleast_1d(tide_ocean['XT'].data[i2])
            # # tide amplitude and phase
            # ph1 = np.broadcast_to(ph['AT'][s,:], (n1,nc))
            # ph2 = np.broadcast_to(ph['XT'][i2,:], (n2,nc))
            # amp1 = np.broadcast_to(amp['AT'][s,:], (n1,nc))
            # amp2 = np.broadcast_to(amp['XT'][i2,:], (n2,nc))

            # combine along-track and across-track variables
            if np.any(i2):
                h = np.concatenate((h1,h2),axis=0)
                t = np.concatenate((t1,t2),axis=0)
                # dt = np.concatenate((dt1,dt2),axis=0)
                tide = np.concatenate((ot1,ot2),axis=0)
                # # calculate complex phase in radians for Euler's
                # cph = -1j*np.concatenate((ph1,ph2),axis=0)*np.pi/180.0
                # # calculate constituent oscillation
                # hc = np.concatenate((amp1,amp2),axis=0)*np.exp(cph)
            else:
                h = np.copy(h1)
                t = np.copy(t1)
                # dt = np.copy(dt1)
                tide = np.copy(ot1)
                # # calculate complex phase in radians for Euler's
                # cph = -1j*ph1*np.pi/180.0
                # # calculate constituent oscillation
                # hc = amp1*np.exp(cph)

            # # convert time to Modified Julian Days (MJD)
            # pu,pf,G = load_nodal_corrections(t + 48622.0, c,
            #     DELTAT=dt, CORRECTIONS=model_format)
            # # cosine and sine components of tidal oscillation
            # ccos = np.zeros_like(t)
            # ssin = np.zeros_like(t)
            # for k,cons in enumerate(c):
            #     if model_format in ('OTIS','ATLAS','netcdf'):
            #         # load parameters for each constituent
            #         _,phase,omega,_,_ = load_constituent(cons)
            #         th = omega*t*86400.0 + phase + pu[:,k]
            #     elif model_format in ('GOT','FES'):
            #         th = G[:,k]*np.pi/180.0 + pu[:,k]
            #     # add to sin and cosine components
            #     ccos += pf[:,k]*hc.real[:,k]*np.cos(th)
            #     ssin -= pf[:,k]*hc.imag[:,k]*np.sin(th)
            # # combined tidal components
            # tide = ccos + ssin

            # create design matrix
            p0 = np.ones_like(t)
            DMAT = np.transpose([p0,t,tide])
            n_max,n_terms = np.shape(DMAT)

            # tuple for parameter bounds (lower and upper)
            lb,ub = ([np.min(h),-10.0,0.0],[np.max(h),10.0,1.0])
            # use linear least-squares with bounds on the variables
            try:
                results = scipy.optimize.lsq_linear(DMAT, h, bounds=(lb,ub))
                # nu = Degrees of Freedom
                nu = n_max - n_terms
                # Mean square error
                MSE = np.sum(results['fun']**2)/np.float(nu)
                # Covariance Matrix
                # Multiplying the design matrix by itself
                Hinv = np.linalg.inv(np.dot(np.transpose(DMAT),DMAT))
                # Taking the diagonal components of the cov matrix
                hdiag = np.diag(Hinv)
                # Default is 95% confidence interval
                alpha = 1.0 - 0.95
                # Student T-Distribution with D.O.F. nu
                # t.ppf parallels tinv in matlab
                tstar = scipy.stats.t.ppf(1.0-(alpha/2.0),nu)
                # beta_err is the error for each coefficient
                # beta_err = t(nu,1-alpha/2)*standard error
                st_err = np.sqrt(MSE*hdiag)
                beta_err = tstar*st_err
            except:
                continue
            else:
                # H,dH,cadj,sadj = np.copy(results['x'])
                H,dH,adj = np.copy(results['x'])
            # extract along-track and across-track cosine and sine
            # tide_ocean['AT'][s,i1] = cadj*ccos[:n1] + sadj*ssin[:n1]
            tide_ocean['AT'][s,i1] = adj*tide[:n1]
            if np.any(i2):
                # tide_ocean['XT'][i2] = cadj*ccos[n1:] + sadj*ssin[n1:]
                tide_ocean['XT'][i2] = adj*tide[n1:]

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

    # output flexure correction HDF5 file
    args = (PRD,TIDE_MODEL,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
    file_format = '{0}_{1}_FIT_TIDES_{2}{3}_{4}{5}_{6}_{7}{8}.h5'
    # print file information
    print('\t{0}'.format(file_format.format(*args))) if VERBOSE else None
    HDF5_ATL11_corr_write(IS2_atl11_tide, IS2_atl11_tide_attrs,
        CLOBBER=True, INPUT=os.path.basename(FILE),
        CROSSOVERS=True, FILL_VALUE=IS2_atl11_fill, DIMENSIONS=IS2_atl11_dims,
        FILENAME=os.path.join(DIRECTORY,file_format.format(*args)))
    # change the permissions mode
    os.chmod(os.path.join(DIRECTORY,file_format.format(*args)), MODE)

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
    leaps = icesat2_toolkit.time.count_leap_seconds(gps_seconds)
    # convert from seconds since 1980-01-06T00:00:00 to Julian days
    MJD = icesat2_toolkit.time.convert_delta_time(gps_seconds - leaps,
        epoch1=(1980,1,6,0,0,0), epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    YY,MM,DD,HH,MN,SS = icesat2_toolkit.time.convert_julian(MJD + 2400000.5,
        FORMAT='tuple')
    # add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = '{0:0.0f}'.format(tmx-tmn)
    # Closing the HDF5 file
    fileID.close()

# Main program that calls fit_tides_ICESat2()
def main():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Calculates ice sheet grounding zones with ICESat-2
            ATL11 annual land ice height data
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
    model_choices = ('CATS0201','CATS2008',
        'TPXO9-atlas','TPXO9-atlas-v2','TPXO9-atlas-v3','TPXO9-atlas-v4',
        'TPXO9.1','TPXO8-atlas','TPXO7.2',
        'AODTM-5','AOTIM-5','AOTIM-5-2018',
        'GOT4.7','GOT4.8','GOT4.10',
        'FES2014')
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, default='CATS2008',
        choices=model_choices,
        help='Tide model to use in correction')
    ib_choices = ['ERA-Interim','ERA5','MERRA-2']
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str, choices=ib_choices,
        help='Reanalysis model to use in inverse-barometer correction')
    # interpolation method
    parser.add_argument('--interpolate','-I',
        metavar='METHOD', type=str, default='spline',
        choices=('spline','linear','nearest','bilinear'),
        help='Spatial interpolation method')
    # verbosity settings
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    args = parser.parse_args()

    # run for each input ATL11 file
    for FILE in args.infile:
        fit_tides_ICESat2(args.directory, FILE, TIDE_MODEL=args.tide,
            REANALYSIS=args.reanalysis, METHOD=args.interpolate,
            VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()