#!/usr/bin/env python
u"""
calculate_GZ_ICESat2_ATL11.py
Written by Tyler Sutterley (05/2024)

Calculates ice sheet grounding zones with ICESat-2 data following:
    Brunt et al., Annals of Glaciology, 51(55), 2010
        https://doi.org/10.3189/172756410791392790
    Fricker et al. Geophysical Research Letters, 33(15), 2006
        https://doi.org/10.1029/2006GL026907
    Fricker et al. Antarctic Science, 21(5), 2009
        https://doi.org/10.1017/S095410200999023X

Outputs an HDF5 file of flexure scaled to match the downstream tide model

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
    --mean-file X: Mean elevation file to remove from the height data
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
    -S, --sea-level: Remove mean dynamic topography from heights
    -C, --crossovers: Run ATL11 Crossovers
    -P, --plot: Create plots of flexural zone
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
    fiona: Python wrapper for vector data access functions from the OGR library
        https://fiona.readthedocs.io/en/latest/manual.html
    shapely: PostGIS-ish operations outside a database context for Python
        http://toblerity.org/shapely/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/ATL11.py: reads ICESat-2 annual land ice height data files
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        use time functions from timescale.time
    Updated 07/2023: using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 11/2022: verify coordinate reference system of shapefile
        added option to remove a static mean file from heights
    Updated 10/2022: made reading mean dynamic topography an option
    Updated 08/2022: use logging for verbose output of processing run
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: use argparse descriptions within documentation
        output estimated elastic modulus in grounding zone data group
    Updated 03/2021: output HDF5 file of flexure scaled by a tide model
        estimate flexure for crossovers using along-track model outputs
        final extent of the flexure AT is the estimated grounding line
        output grounding zone data group to output fit statistics
        replaced numpy bool/int to prevent deprecation warnings
        use utilities to set default path to shapefiles
    Updated 01/2021: using standalone ATL11 reader
        using argparse to set command line options
        using time module for conversion operations
    Written 12/2020
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import datetime
import argparse
import operator
import itertools
import traceback
import numpy as np
import collections
import scipy.stats
import scipy.optimize
import grounding_zones as gz

# attempt imports
fiona = gz.utilities.import_dependency('fiona')
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
plt = gz.utilities.import_dependency('matplotlib.pyplot')
pyproj = gz.utilities.import_dependency('pyproj')
geometry = gz.utilities.import_dependency('shapely.geometry')
timescale = gz.utilities.import_dependency('timescale')

# grounded ice shapefiles
grounded_shapefile = {}
grounded_shapefile['N'] = 'grn_ice_sheet_peripheral_glaciers.shp'
grounded_shapefile['S'] = 'ant_ice_sheet_islands_v2.shp'
# description and reference for each grounded ice file
grounded_description = {}
grounded_description['N'] = 'Greenland Mapping Project (GIMP) Ice & Ocean Mask'
grounded_description['S'] = ('MEaSUREs Antarctic Boundaries for IPY 2007-2009 '
    'from Satellite_Radar, Version 2')
grounded_reference = {}
grounded_reference['N'] = 'https://doi.org/10.5194/tc-8-1509-2014'
grounded_reference['S'] = 'https://doi.org/10.5067/IKBWW4RYHF1Q'

# PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        projection_flag = 'S'
    elif GRANULE in ('03','04','05'):
        projection_flag = 'N'
    return projection_flag

# PURPOSE: find if segment crosses previously-known grounding line position
def read_grounded_ice(base_dir, HEM, VARIABLES=[0]):
    # reading grounded ice shapefile
    input_shapefile = base_dir.joinpath(grounded_shapefile[HEM])
    shape = fiona.open(str(input_shapefile))
    # extract coordinate reference system
    if ('init' in shape.crs.keys()):
        epsg = pyproj.CRS(shape.crs['init']).to_epsg()
    else:
        epsg = pyproj.CRS(shape.crs).to_epsg()
    # reduce to variables of interest if specified
    shape_entities = [f for f in shape.values() if int(f['id']) in VARIABLES]
    # create list of polygons
    lines = []
    # extract the entities and assign by tile name
    for i,ent in enumerate(shape_entities):
        # extract coordinates for entity
        line_obj = geometry.LineString(ent['geometry']['coordinates'])
        lines.append(line_obj)
    # create shapely multilinestring object
    mline_obj = geometry.MultiLineString(lines)
    # close the shapefile
    shape.close()
    # return the line string object for the ice sheet
    return (mline_obj, epsg)

# PURPOSE: Find indices of common reference points between two lists
# Determines which along-track points correspond with the across-track
def common_reference_points(XT, AT):
    ind2 = np.squeeze([np.flatnonzero(AT == p) for p in XT])
    return ind2

# PURPOSE: compress complete list of valid indices into a set of ranges
def compress_list(i,n):
    for a,b in itertools.groupby(enumerate(i), lambda v: ((v[1]-v[0])//n)*n):
        group = list(map(operator.itemgetter(1),b))
        yield (group[0], group[-1])

# PURPOSE: read ICESat-2 annual land ice height data (ATL11) from NSIDC
# calculate mean elevation between all dates in file
# calculate inflexion point using elevation surface slopes
# use mean elevation to calculate elevation anomalies
# use anomalies to calculate inward and seaward limits of tidal flexure
def calculate_GZ_ICESat2(base_dir, INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    CROSSOVERS=False,
    MEAN_FILE=None,
    TIDE_MODEL=None,
    REANALYSIS=None,
    SEA_LEVEL=False,
    PLOT=False,
    MODE=0o775):

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(GRANULE).pop()
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # file format for associated auxiliary files
    file_format = '{0}_{1}_{2}_{3}{4}_{5}{6}_{7}_{8}{9}.h5'
    # set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRAN)

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = gz.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = gz.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()

    # read data from input ATL11 file
    mds1,attr1,pairs1 = is2tk.io.ATL11.read_granule(INPUT_FILE,
        REFERENCE=True, CROSSOVERS=CROSSOVERS, ATTRIBUTES=True)

    # grounded ice line string to determine if segment crosses coastline
    mline_obj, epsg = read_grounded_ice(base_dir, HEM, VARIABLES=[0])

    # height threshold (filter points below 0m elevation)
    THRESHOLD = 0.0
    # densities of seawater and ice
    rho_w = 1030.0
    rho_ice = 917.0

    # projections for converting lat/lon to polar stereographic
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(epsg)
    # transformer object for converting projections
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # copy variables for outputting to HDF5 file
    IS2_atl11_gz = {}
    IS2_atl11_fill = {}
    IS2_atl11_dims = {}
    IS2_atl11_gz_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl11_gz['ancillary_data'] = {}
    IS2_atl11_gz_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl11_gz['ancillary_data'][key] = mds1['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl11_gz_attrs['ancillary_data'][key] = {}
        for att_name,att_val in attr1['ancillary_data'][key].items():
            IS2_atl11_gz_attrs['ancillary_data'][key][att_name] = att_val
    # HDF5 group name for across-track data
    XT = 'crossing_track_data'
    # HDF5 group name for grounding zone data
    GZD = 'grounding_zone_data'
    GROUNDING_ZONE = True

    # for each input beam within the file
    for ptx in sorted(pairs1):
        # output data dictionaries for beam pair
        IS2_atl11_gz[ptx] = dict(cycle_stats=collections.OrderedDict(),
            crossing_track_data=collections.OrderedDict(),
            grounding_zone_data=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})
        IS2_atl11_dims[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})
        IS2_atl11_gz_attrs[ptx] = dict(cycle_stats={},crossing_track_data={},
            grounding_zone_data={})

        # extract along-track and across-track variables
        ref_pt = {}
        latitude = {}
        longitude = {}
        delta_time = {}
        h_corr = {}
        tide_ocean = {}
        IB = {}
        groups = ['AT']
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
        quality_summary = (mds1[ptx]['quality_summary'] == 0)
        # ocean corrections
        tide_ocean['AT'] = np.ma.array(mds1[ptx]['cycle_stats']['tide_ocean'],
            fill_value=attr1[ptx]['cycle_stats']['tide_ocean']['_FillValue'])
        tide_ocean['AT'].mask = (tide_ocean['AT'] == tide_ocean['AT'].fill_value)
        IB['AT'] = np.ma.array(mds1[ptx]['cycle_stats']['dac'],
            fill_value=attr1[ptx]['cycle_stats']['dac']['_FillValue'])
        IB['AT'].mask = (IB['AT'] == IB['AT'].fill_value)
        # ATL11 geoid height values
        geoid_h = mds1[ptx]['ref_surf']['geoid_h']
        # if running ATL11 crossovers
        if CROSSOVERS:
            # add to group
            groups.append('XT')
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
            IB['XT'] = np.ma.array(mds1[ptx][XT]['dac'],
                fill_value=attr1[ptx][XT]['dac']['_FillValue'])
            IB['XT'].mask = (IB['XT'] == IB['XT'].fill_value)

        # read buffered grounding zone mask
        a2 = (PRD,'GROUNDING_ZONE','MASK',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
        f3 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a2))
        # create data mask for grounding zone
        mds1[ptx]['subsetting'] = {}
        mds1[ptx]['subsetting'].setdefault('ice_gz',
            np.zeros((n_points),dtype=bool))
        # check that mask file exists
        try:
            mds2,attr2 = is2tk.io.ATL11.read_pair(f3, ptx,
                ATTRIBUTES=True, VERBOSE=False, SUBSETTING=True)
        except Exception as exc:
            logging.debug(traceback.format_exc())
            pass
        else:
            mds1[ptx]['subsetting']['ice_gz'] = \
                mds2[ptx]['subsetting']['ice_gz']
            B = attr2[ptx]['subsetting']['ice_gz']['source']

        # read mean elevation file (e.g. digital elevation model)
        dem_h = np.ma.zeros((n_points), fill_value=fv)
        if MEAN_FILE:
            # read DEM HDF5 file
            try:
                mds2,attr2 = is2tk.io.ATL11.read_pair(MEAN_FILE, ptx,
                    REFERENCE=True, VERBOSE=False)
                dem_h.data[:] = mds2[ptx]['ref_surf']['dem_h'].copy()
                fv2 = attr2[ptx]['dem']['ref_surf']['_FillValue']
            except Exception as exc:
                logging.debug(traceback.format_exc())
                dem_h.mask = np.ones((n_points),dtype=bool)
            else:
                dem_h.mask = (dem_h.data[:] == fv2)
        else:
            # use default DEM within ATL11
            # ATL11 reference surface elevations are derived from ATL06
            dem_h.data[:] = mds1[ptx]['ref_surf']['dem_h'].copy()
            fv2 = attr1[ptx]['ref_surf']['dem_h']['_FillValue']
            dem_h.mask = (dem_h.data[:] == fv2)

        # read tide model
        if TIDE_MODEL:
            # read tide model HDF5 file
            a3 = (PRD,TIDE_MODEL,'TIDES',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            f3 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a3))
            # check that tide model file exists
            try:
                mds3,attr3 = is2tk.io.ATL11.read_pair(f3, ptx,
                    VERBOSE=False, CROSSOVERS=CROSSOVERS)
            except Exception as exc:
                logging.debug(traceback.format_exc())
                # mask all values
                for group in groups:
                    tide_ocean[group].mask[:] = True
                pass
            else:
                tide_ocean['AT'].data[:] = mds3[ptx]['cycle_stats']['tide_ocean']
                if CROSSOVERS:
                    tide_ocean['XT'].data[:] = mds3[ptx][XT]['tide_ocean']
            # source of tide model
            tide_source = TIDE_MODEL
        else:
            tide_source = 'ATL06'
        # set masks and fill values
        for group,val in tide_ocean.items():
            val.mask[:] = (val.data == val.fill_value)
            val.mask[:] |= (h_corr[group].data == h_corr[group].fill_value)
            val.data[val.mask] = val.fill_value

        # read inverse barometer correction
        if REANALYSIS:
            # read inverse barometer HDF5 file
            a4 = (PRD,REANALYSIS,'IB',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            f4 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a4))
            # check that inverse barometer file exists
            try:
                mds4,attr4 = is2tk.io.ATL11.read_pair(f4,ptx,
                    VERBOSE=False,CROSSOVERS=CROSSOVERS)
            except Exception as exc:
                logging.debug(traceback.format_exc())
                # mask all values
                for group in groups:
                    IB[group].mask[:] = True
                pass
            else:
                IB['AT'].data[:] = mds4[ptx]['cycle_stats']['ib']
                if CROSSOVERS:
                    IB['XT'].data[:] = mds4[ptx][XT]['ib']
        # set masks and fill values
        for group,val in IB.items():
            val.mask[:] = (val.data == val.fill_value)
            val.mask[:] |= (h_corr[group].data == h_corr[group].fill_value)
            val.data[val.mask] = val.fill_value

        # mean dynamic topography
        if SEA_LEVEL:
            a5 = (PRD,'AVISO','SEA_LEVEL',TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            f5 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a5))
            # check that mean dynamic topography file exists
            try:
                mds5,attr5 = is2tk.io.ATL11.read_pair(f5, ptx,
                    VERBOSE=False)
            except Exception as exc:
                logging.debug(traceback.format_exc())
                mdt = np.zeros((n_points))
                pass
            else:
                mdt = mds5[ptx]['cycle_stats']['mdt']

        # extract lat/lon and convert to polar stereographic
        X,Y = transformer.transform(longitude['AT'],latitude['AT'])
        # along-track (AT) flexure corrections
        fv = attr1[ptx]['cycle_stats']['tide_ocean']['_FillValue']
        flexure = np.ma.zeros((n_points,n_cycles),fill_value=fv)
        # initally copy the ocean tide estimate
        flexure.data[:] = np.copy(tide_ocean['AT'].data)
        flexure.mask = np.copy(tide_ocean['AT'].mask)
        # scaling factor for segment tides
        scaling = np.ma.ones((n_points,n_cycles),fill_value=0.0)
        scaling.mask = np.copy(tide_ocean['AT'].mask)
        scaling.data[scaling.mask]

        # outputs of grounding zone fit
        grounding_zone_data = {}
        grounding_zone_data['ref_pt'] = []
        grounding_zone_data['latitude'] = []
        grounding_zone_data['longitude'] = []
        grounding_zone_data['delta_time'] = []
        grounding_zone_data['cycle_number'] = []
        # grounding_zone_data['tide_ocean'] = []
        grounding_zone_data['gz_sigma'] = []
        grounding_zone_data['e_mod'] = []
        grounding_zone_data['e_mod_sigma'] = []
        # grounding_zone_data['H_ice'] = []
        # grounding_zone_data['delta_h'] = []

        # if creating a test plot
        valid_plot = False
        if PLOT:
            f1,ax1 = plt.subplots(num=1,figsize=(13,7))

        # for each cycle of ATL11 data
        for c,CYCLE in enumerate(mds1[ptx]['cycle_number']):
            # find valid points with GZ for any ATL11 cycle
            segment_mask = np.logical_not(h_corr['AT'].mask[:,c])
            segment_mask = np.logical_not(tide_ocean['AT'].mask[:,c])
            segment_mask &= (h_corr['AT'].data[:,c] > THRESHOLD)
            segment_mask &= mds1[ptx]['subsetting']['ice_gz']
            segment_mask &= quality_summary[:,c]
            ifit, = np.nonzero(segment_mask)
            # segment of points within grounding zone
            igz, = np.nonzero(mds1[ptx]['subsetting']['ice_gz'])

            # compress list (separate geosegs into sets of ranges)
            ice_gz_indices = compress_list(ifit,1000)
            for imin,imax in ice_gz_indices:
                # find valid indices within range
                i = sorted(set(np.arange(imin,imax+1)) & set(ifit))
                iout = sorted(set(np.arange(imin,imax+1)) & set(igz))
                coords = np.sqrt((X-X[i[0]])**2 + (Y-Y[i[0]])**2)
                # shapely LineString object for altimetry segment
                try:
                    segment_line = geometry.LineString(np.c_[X[i], Y[i]])
                except:
                    continue
                # determine if line segment intersects previously known GZ
                if segment_line.intersects(mline_obj):
                    # extract intersected point (find minimum distance)
                    try:
                        xi,yi = mline_obj.intersection(segment_line).xy
                    except:
                        continue
                    else:
                        iint = np.argmin((Y[i]-yi)**2 + (X[i]-xi)**2)
                    # horizontal eulerian distance from start of segment
                    dist = coords[i]
                    output = coords[iout]
                    # land ice height for grounding zone
                    h_gz = np.copy(h_corr['AT'].data[i,c])
                    # mean land ice height from digital elevation model
                    h_mean = np.mean(h_corr['AT'][i,:],axis=1)
                    # h_mean = h_corr['AT'].data[i,0]
                    # ocean tide height for scaling model
                    tide_mean =  np.mean(tide_ocean['AT'][i,:],axis=1)
                    # tide_mean = tide_ocean['AT'].data[i,0]
                    h_tide = np.ma.array(tide_ocean['AT'].data[i,c] - tide_mean,
                        fill_value=tide_ocean['AT'].fill_value)
                    h_tide.mask = tide_ocean['AT'].mask[i,c] | tide_mean.mask
                    # inverse-barometer response
                    ib_mean =  np.mean(IB['AT'][i,:],axis=1)
                    # ib_mean = IB['AT'].data[i,0]
                    h_ib = np.ma.array(IB['AT'].data[i,c] - ib_mean,
                        fill_value=IB['AT'].fill_value)
                    h_ib.mask = IB['AT'].mask[i,c] | ib_mean.mask
                    # deflection from mean land ice height in grounding zone
                    dh_gz = h_gz - h_mean
                    # quasi-freeboard: WGS84 elevation - geoid height
                    QFB = h_gz - geoid_h[i]
                    # ice thickness from quasi-freeboard and densities
                    w_thick = QFB*rho_w/(rho_w-rho_ice)
                    # fit with a hard piecewise model to get rough estimate of GZ
                    try:
                        C1, C2, PWMODEL = gz.fit.piecewise_bending(
                            dist, dh_gz, STEP=5)
                    except:
                        continue
                    # distance from estimated grounding line (0 = grounding line)
                    d = (dist - C1[0]).astype(int)
                    # determine if spacecraft is approaching coastline
                    sco = True if np.mean(h_gz[d<0]) < np.mean(h_gz[d>0]) else False
                    # set initial fit outputs to infinite
                    GZ = np.array([np.inf, np.inf])
                    PGZ = np.array([np.inf, np.inf])
                    # set grounding zone estimates for testing
                    GRZ = []
                    # 1,2: use GZ location values from piecewise fit
                    # 3,4: use GZ location values from known grounding line
                    GRZ.append(C1)
                    GRZ.append(C1)
                    GRZ.append([dist[iint],dist[iint]-2e3,dist[iint]+2e3])
                    GRZ.append([dist[iint],dist[iint]-2e3,dist[iint]+2e3])
                    # set tide values for testing
                    TIDE = []
                    i0 = 0 if sco else -1
                    tplus = h_tide[i0] + h_ib[i0]
                    # 1,3: use tide range values from Padman (2002)
                    # 2,4: use tide range values from model+ib
                    TIDE.append([1.2,-3.0,3.0])
                    TIDE.append([tplus,tplus-0.3,tplus+0.3])
                    TIDE.append([1.2,-3.0,3.0])
                    TIDE.append([tplus,tplus-0.3,tplus+0.3])
                    # iterate through tests
                    for grz,tide in zip(GRZ,TIDE):
                        # fit physical elastic model
                        try:
                            GZ,PA,PE,PT,PdH,MODEL = gz.fit.elastic_bending(
                                dist, dh_gz, GRZ=grz, TIDE=tide,
                                ORIENTATION=sco, THICKNESS=w_thick,
                                CONF=0.95, XOUT=output)
                        except:
                            pass
                        # copy grounding zone parameters to get best fit
                        if (GZ[1] < PGZ[1]):
                            PGZ = np.copy(GZ)
                            model_scale = np.copy(PA[0])
                            PEMODEL = np.copy(MODEL)
                        # use parameters if fit significance is within tolerance
                        if (GZ[1] < 400.0):
                            break
                    # skip saving parameters if no valid solution was found
                    if np.isnan(PGZ[0]):
                        continue

                    # linearly interpolate distance to grounding line
                    GZrpt = np.interp(PGZ[0],output,ref_pt['AT'][iout])
                    GZlat = np.interp(PGZ[0],output,latitude['AT'][iout])
                    GZlon = np.interp(PGZ[0],output,longitude['AT'][iout])
                    GZtime = np.interp(PGZ[0],dist,delta_time['AT'][i,c])
                    # append outputs of grounding zone fit
                    # save all outputs (not just within tolerance)
                    grounding_zone_data['ref_pt'].append(GZrpt)
                    grounding_zone_data['latitude'].append(GZlat)
                    grounding_zone_data['longitude'].append(GZlon)
                    grounding_zone_data['delta_time'].append(GZtime)
                    grounding_zone_data['cycle_number'].append(CYCLE)
                    # grounding_zone_data['tide_ocean'].append(PA)
                    grounding_zone_data['gz_sigma'].append(PGZ[1])
                    grounding_zone_data['e_mod'].append(PE[0]/1e9)
                    grounding_zone_data['e_mod_sigma'].append(PE[1]/1e9)
                    # grounding_zone_data['H_ice'].append(PT)
                    # grounding_zone_data['delta_h'].append(PdH)

                    # reorient input parameters to go from land ice to floating
                    flexure_mask = np.ones_like(iout,dtype=bool)
                    if sco:
                        # start of segment in orientation
                        i0 = iout[0]
                        # mean tide for scaling and plots
                        # mean_tide = tide_ocean['AT'].data[i0,0]
                        mean_tide = np.mean(tide_ocean['AT'][i0,:])
                        mean_ib = np.mean(IB['AT'][i0,:])
                        tide_scale = tide_ocean['AT'].data[i0,c] - mean_tide
                        # replace mask values for points beyond the grounding line
                        ii, = np.nonzero(ref_pt['AT'][iout] <= GZrpt)
                        flexure_mask[ii] = False
                    else:
                        # start of segment in orientation
                        i0 = iout[-1]
                        # mean tide for scaling and plots
                        # mean_tide = tide_ocean['AT'].data[i0,0]
                        mean_tide = np.mean(tide_ocean['AT'][i0,:])
                        mean_ib = np.mean(IB['AT'][i0,:])
                        tide_scale = tide_ocean['AT'].data[i0,c] - mean_tide
                        # replace mask values for points beyond the grounding line
                        ii, = np.nonzero(ref_pt['AT'][iout] >= GZrpt)
                        flexure_mask[ii] = False
                    # add to test plot
                    if PLOT:
                        # plot height differences
                        l, = ax1.plot(ref_pt['AT'][i],dh_gz-PdH[0],'.-',ms=1.5,lw=0,
                            label=f'Cycle {CYCLE}')
                        # plot downstream tide and IB
                        hocean = tide_ocean['AT'].data[i0,c] - mean_tide
                        # hocean += IB['AT'].data[i0,c] - mean_ib
                        ax1.axhline(hocean,color=l.get_color(),lw=3.0,ls='--')
                        # set valid plot flag
                        valid_plot = True

                    # if the grounding zone errors are not within tolerance
                    if (PGZ[1] >= 800.0):
                        # leave iteration and keep original tide model
                        # for segment
                        continue

                    # calculate scaling factor
                    scale_factor = tide_scale/model_scale
                    # scale flexure and restore mean ocean tide
                    flexure[iout,c] = scale_factor*(PEMODEL-PdH[0]) + mean_tide
                    flexure.mask[iout,c] = flexure_mask
                    # scaling factor between current tide and flexure
                    scaling[iout,c] = flexure[iout,c]/tide_ocean['AT'].data[iout,c]
                    scaling.mask[iout,c] = flexure_mask
                    # add to test plot
                    if PLOT:
                        # plot elastic deformation model
                        ax1.plot(ref_pt['AT'][iout],PEMODEL-PdH[0],
                            color='0.3',lw=2,zorder=9)
                        # plot scaled elastic deformation model
                        ax1.plot(ref_pt['AT'][iout],flexure[iout,c]-mean_tide,
                            color='0.8',lw=2,zorder=10)
                        # plot grounding line location
                        ax1.axvline(GZrpt,color=l.get_color(),
                            ls='--',dashes=(8,4))

        # make final plot adjustments and save to file
        if valid_plot:
            # add legend
            lgd = ax1.legend(loc=1,frameon=True)
            # set width, color and style of lines
            lgd.get_frame().set_boxstyle('square,pad=0.1')
            lgd.get_frame().set_edgecolor('white')
            lgd.get_frame().set_alpha(1.0)
            for line,text in zip(lgd.get_lines(),lgd.get_texts()):
                line.set_linewidth(6)
                text.set_weight('bold')
                text.set_color(line.get_color())
            # adjust figure
            f1.subplots_adjust(left=0.05,right=0.97,bottom=0.04,top=0.96,hspace=0.15)
            # create plot file of flexural zone
            args = (PRD,ptx,TIDE_MODEL,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
            plot_format = '{0}_{1}_{2}_GZ_TIDES_{3}{4}_{5}{6}_{7}_{8}{9}.png'
            output_plot_file = OUTPUT_DIRECTORY.joinpath(plot_format.format(*args))
            # log output plot file
            logging.info(str(output_plot_file))
            f1.savefig(output_plot_file, dpi=240, format='png',
                metadata={'Title':pathlib.Path(sys.argv[0]).name})
            # clear all figure axes
            plt.cla()
            plt.clf()

        # group attributes for beam
        IS2_atl11_gz_attrs[ptx]['description'] = ('Contains the primary science parameters '
            'for this data set')
        IS2_atl11_gz_attrs[ptx]['beam_pair'] = attr1[ptx]['beam_pair']
        IS2_atl11_gz_attrs[ptx]['ReferenceGroundTrack'] = attr1[ptx]['ReferenceGroundTrack']
        IS2_atl11_gz_attrs[ptx]['first_cycle'] = attr1[ptx]['first_cycle']
        IS2_atl11_gz_attrs[ptx]['last_cycle'] = attr1[ptx]['last_cycle']
        IS2_atl11_gz_attrs[ptx]['equatorial_radius'] = attr1[ptx]['equatorial_radius']
        IS2_atl11_gz_attrs[ptx]['polar_radius'] = attr1[ptx]['polar_radius']

        # geolocation, time and reference point
        # reference point
        IS2_atl11_gz[ptx]['ref_pt'] = ref_pt['AT'].copy()
        IS2_atl11_fill[ptx]['ref_pt'] = None
        IS2_atl11_dims[ptx]['ref_pt'] = None
        IS2_atl11_gz_attrs[ptx]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['ref_pt']['units'] = "1"
        IS2_atl11_gz_attrs[ptx]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_gz_attrs[ptx]['ref_pt']['long_name'] = "Reference point number"
        IS2_atl11_gz_attrs[ptx]['ref_pt']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['ref_pt']['description'] = ("The reference point is the "
            "7 digit segment_id number corresponding to the center of the ATL06 data used "
            "for each ATL11 point.  These are sequential, starting with 1 for the first "
            "segment after an ascending equatorial crossing node.")
        IS2_atl11_gz_attrs[ptx]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number
        IS2_atl11_gz[ptx]['cycle_number'] = mds1[ptx]['cycle_number'].copy()
        IS2_atl11_fill[ptx]['cycle_number'] = None
        IS2_atl11_dims[ptx]['cycle_number'] = None
        IS2_atl11_gz_attrs[ptx]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['cycle_number']['units'] = "1"
        IS2_atl11_gz_attrs[ptx]['cycle_number']['long_name'] = "Orbital cycle number"
        IS2_atl11_gz_attrs[ptx]['cycle_number']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['cycle_number']['description'] = ("Number of 91-day periods "
            "that have elapsed since ICESat-2 entered the science orbit. Each of the 1,387 "
            "reference ground track (RGTs) is targeted in the polar regions once "
            "every 91 days.")
        # delta time
        IS2_atl11_gz[ptx]['delta_time'] = delta_time['AT'].copy()
        IS2_atl11_fill[ptx]['delta_time'] = delta_time['AT'].fill_value
        IS2_atl11_dims[ptx]['delta_time'] = ['ref_pt','cycle_number']
        IS2_atl11_gz_attrs[ptx]['delta_time'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_gz_attrs[ptx]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_gz_attrs[ptx]['delta_time']['standard_name'] = "time"
        IS2_atl11_gz_attrs[ptx]['delta_time']['calendar'] = "standard"
        IS2_atl11_gz_attrs[ptx]['delta_time']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_gz_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt cycle_number latitude longitude"
        # latitude
        IS2_atl11_gz[ptx]['latitude'] = latitude['AT'].copy()
        IS2_atl11_fill[ptx]['latitude'] = latitude['AT'].fill_value
        IS2_atl11_dims[ptx]['latitude'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx]['latitude'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['latitude']['units'] = "degrees_north"
        IS2_atl11_gz_attrs[ptx]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx]['latitude']['long_name'] = "Latitude"
        IS2_atl11_gz_attrs[ptx]['latitude']['standard_name'] = "latitude"
        IS2_atl11_gz_attrs[ptx]['latitude']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_gz_attrs[ptx]['latitude']['valid_min'] = -90.0
        IS2_atl11_gz_attrs[ptx]['latitude']['valid_max'] = 90.0
        IS2_atl11_gz_attrs[ptx]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude
        IS2_atl11_gz[ptx]['longitude'] = longitude['AT'].copy()
        IS2_atl11_fill[ptx]['longitude'] = longitude['AT'].fill_value
        IS2_atl11_dims[ptx]['longitude'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx]['longitude'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['longitude']['units'] = "degrees_east"
        IS2_atl11_gz_attrs[ptx]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx]['longitude']['long_name'] = "Longitude"
        IS2_atl11_gz_attrs[ptx]['longitude']['standard_name'] = "longitude"
        IS2_atl11_gz_attrs[ptx]['longitude']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_gz_attrs[ptx]['longitude']['valid_min'] = -180.0
        IS2_atl11_gz_attrs[ptx]['longitude']['valid_max'] = 180.0
        IS2_atl11_gz_attrs[ptx]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"

        # cycle statistics variables
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['Description'] = ("The cycle_stats subgroup "
            "contains summary information about segments for each reference point, including "
            "the uncorrected mean heights for reference surfaces, blowing snow and cloud "
            "indicators, and geolocation and height misfit statistics.")
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['data_rate'] = ("Data within this group "
            "are stored at the average segment rate.")
        # computed tide with flexure
        flexure.data[flexure.mask] = flexure.fill_value
        IS2_atl11_gz[ptx]['cycle_stats']['tide_ocean'] = flexure.copy()
        IS2_atl11_fill[ptx]['cycle_stats']['tide_ocean'] = flexure.fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['tide_ocean'] = ['ref_pt','cycle_number']
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['units'] = "meters"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['contentType'] = "referenceInformation"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['long_name'] = "Ocean Tide"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['description'] = ("Ocean Tides with "
            "Near-Grounding Zone Flexure that includes diurnal and semi-diurnal (harmonic analysis), "
            "and longer period tides (dynamic and self-consistent equilibrium).")
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['source'] = tide_source
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['reference'] = \
            "https://doi.org/10.3189/172756410791392790"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['tide_ocean']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"
        # ratio of flexure with respect to downstream ocean tide
        scaling.data[scaling.mask] = scaling.fill_value
        IS2_atl11_gz[ptx]['cycle_stats']['flexure'] = scaling.copy()
        IS2_atl11_fill[ptx]['cycle_stats']['flexure'] = scaling.fill_value
        IS2_atl11_dims[ptx]['cycle_stats']['flexure'] = ['ref_pt','cycle_number']
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['units'] = "1"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['contentType'] = "referenceInformation"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['long_name'] = "Flexure Ratio"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['description'] = ("Ratio of "
            "Near-Grounding Zone Flexure with respect to Downstream Ocean Tide Height.")
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['source'] = tide_source
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['reference'] = \
            "https://doi.org/10.3189/172756410791392790"
        IS2_atl11_gz_attrs[ptx]['cycle_stats']['flexure']['coordinates'] = \
            "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"

        # grounding zone variables
        IS2_atl11_gz_attrs[ptx][GZD]['Description'] = ("The grounding_zone_data "
            "subgroup contains statistic data at grounding zone locations.")
        IS2_atl11_gz_attrs[ptx][GZD]['data_rate'] = ("Data within this group are "
            "stored at the average segment rate.")

        # reference point of the grounding zone
        IS2_atl11_gz[ptx][GZD]['ref_pt'] = np.copy(grounding_zone_data['ref_pt'])
        IS2_atl11_fill[ptx][GZD]['ref_pt'] = None
        IS2_atl11_dims[ptx][GZD]['ref_pt'] = None
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['units'] = "1"
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['long_name'] = ("fit center reference point number, "
            "segment_id")
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['source'] = "derived, ATL11 algorithm"
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['description'] = ("The reference-point number of the "
            "fit center for the datum track. The reference point is the 7 digit segment_id number "
            "corresponding to the center of the ATL06 data used for each ATL11 point.  These are "
            "sequential, starting with 1 for the first segment after an ascending equatorial "
            "crossing node.")
        IS2_atl11_gz_attrs[ptx][GZD]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number of the grounding zone
        IS2_atl11_gz[ptx][GZD]['cycle_number'] = np.copy(grounding_zone_data['cycle_number'])
        IS2_atl11_fill[ptx][GZD]['cycle_number'] = None
        IS2_atl11_dims[ptx][GZD]['cycle_number'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number']['units'] = "1"
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number']['long_name'] = "cycle number"
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx][GZD]['cycle_number']['description'] = ("Cycle number for the "
            "grounding zone data. Number of 91-day periods that have elapsed since ICESat-2 entered "
            "the science orbit. Each of the 1,387 reference ground track (RGTs) is targeted "
            "in the polar regions once every 91 days.")
        # delta time of the grounding zone
        IS2_atl11_gz[ptx][GZD]['delta_time'] = np.copy(grounding_zone_data['delta_time'])
        IS2_atl11_fill[ptx][GZD]['delta_time'] = delta_time['AT'].fill_value
        IS2_atl11_dims[ptx][GZD]['delta_time'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time'] = {}
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['standard_name'] = "time"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['calendar'] = "standard"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_gz_attrs[ptx][GZD]['delta_time']['coordinates'] = \
            "ref_pt latitude longitude"
        # latitude of the grounding zone
        IS2_atl11_gz[ptx][GZD]['latitude'] = np.copy(grounding_zone_data['latitude'])
        IS2_atl11_fill[ptx][GZD]['latitude'] = latitude['AT'].fill_value
        IS2_atl11_dims[ptx][GZD]['latitude'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['latitude'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['units'] = "degrees_north"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['long_name'] = "grounding zone latitude"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['standard_name'] = "latitude"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['description'] = ("Center latitude of "
            "the grounding zone")
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['valid_min'] = -90.0
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['valid_max'] = 90.0
        IS2_atl11_gz_attrs[ptx][GZD]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude of the grounding zone
        IS2_atl11_gz[ptx][GZD]['longitude'] = np.copy(grounding_zone_data['longitude'])
        IS2_atl11_fill[ptx][GZD]['longitude'] = longitude['AT'].fill_value
        IS2_atl11_dims[ptx][GZD]['longitude'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['longitude'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['units'] = "degrees_east"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['long_name'] = "grounding zone longitude"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['standard_name'] = "longitude"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['source'] = "ATL06"
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['description'] = ("Center longitude of "
            "the grounding zone")
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['valid_min'] = -180.0
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['valid_max'] = 180.0
        IS2_atl11_gz_attrs[ptx][GZD]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"
        # uncertainty of the grounding zone
        IS2_atl11_gz[ptx][GZD]['gz_sigma'] = np.copy(grounding_zone_data['gz_sigma'])
        IS2_atl11_fill[ptx][GZD]['gz_sigma'] = 0.0
        IS2_atl11_dims[ptx][GZD]['gz_sigma'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['units'] = "meters"
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['long_name'] = "grounding zone uncertainty"
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['source'] = "ATL11"
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['description'] = ("Uncertainty in grounding"
            "zone location derived by the physical elastic bending model")
        IS2_atl11_gz_attrs[ptx][GZD]['gz_sigma']['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # effective elastic modulus
        IS2_atl11_gz[ptx][GZD]['e_mod'] = np.copy(grounding_zone_data['e_mod'])
        IS2_atl11_fill[ptx][GZD]['e_mod'] = 0.0
        IS2_atl11_dims[ptx][GZD]['e_mod'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['units'] = "GPa"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['long_name'] = "Elastic modulus"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['source'] = "ATL11"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['description'] = ("Effective Elastic modulus "
            "of ice estimating using an elastic beam model")
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod']['coordinates'] = \
            "ref_pt delta_time latitude longitude"
        # uncertainty of the elastic modulus
        IS2_atl11_gz[ptx][GZD]['e_mod_sigma'] = np.copy(grounding_zone_data['e_mod_sigma'])
        IS2_atl11_fill[ptx][GZD]['e_mod_sigma'] = 0.0
        IS2_atl11_dims[ptx][GZD]['e_mod_sigma'] = ['ref_pt']
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma'] = collections.OrderedDict()
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['units'] = "GPa"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['contentType'] = "physicalMeasurement"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['long_name'] = "Elastic modulus uncertainty"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['source'] = "ATL11"
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['description'] = ("Uncertainty in the "
            "effective Elastic modulus of ice")
        IS2_atl11_gz_attrs[ptx][GZD]['e_mod_sigma']['coordinates'] = \
            "ref_pt delta_time latitude longitude"

        # if estimating flexure for crossover measurements
        if CROSSOVERS:
            # calculate mean scaling for crossovers
            mean_scale = np.ma.zeros((n_points),fill_value=scaling.fill_value)
            mean_scale.data[:] = scaling.mean(axis=1)
            mean_scale.mask = np.all(scaling.mask,axis=1)
            # find mapping between crossover and along-track reference points
            ref_indices = common_reference_points(ref_pt['XT'], ref_pt['AT'])
            # scale input tide model for estimated flexure in region
            scaled_tide = np.ma.zeros((n_cross),fill_value=tide_ocean['XT'].fill_value)
            scaled_tide.data[:] = tide_ocean['XT']*mean_scale[ref_indices]
            scaled_tide.mask = mean_scale.mask[ref_indices]

            # crossing track variables
            IS2_atl11_gz_attrs[ptx][XT]['Description'] = ("The crossing_track_data "
                "subgroup contains elevation data at crossover locations. These are "
                "locations where two ICESat-2 pair tracks cross, so data are available "
                "from both the datum track, for which the granule was generated, and "
                "from the crossing track.")
            IS2_atl11_gz_attrs[ptx][XT]['data_rate'] = ("Data within this group are "
                "stored at the average segment rate.")

            # reference point
            IS2_atl11_gz[ptx][XT]['ref_pt'] = mds1[ptx][XT]['ref_pt'].copy()
            IS2_atl11_fill[ptx][XT]['ref_pt'] = None
            IS2_atl11_dims[ptx][XT]['ref_pt'] = None
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['units'] = "1"
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['contentType'] = "referenceInformation"
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['long_name'] = ("fit center reference point number, "
                "segment_id")
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['source'] = "derived, ATL11 algorithm"
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['description'] = ("The reference-point number of the "
                "fit center for the datum track. The reference point is the 7 digit segment_id number "
                "corresponding to the center of the ATL06 data used for each ATL11 point.  These are "
                "sequential, starting with 1 for the first segment after an ascending equatorial "
                "crossing node.")
            IS2_atl11_gz_attrs[ptx][XT]['ref_pt']['coordinates'] = \
                "delta_time latitude longitude"
            # reference ground track of the crossing track
            IS2_atl11_gz[ptx][XT]['rgt'] = mds1[ptx][XT]['rgt'].copy()
            IS2_atl11_fill[ptx][XT]['rgt'] = attr1[ptx][XT]['rgt']['_FillValue']
            IS2_atl11_dims[ptx][XT]['rgt'] = None
            IS2_atl11_gz_attrs[ptx][XT]['rgt'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['units'] = "1"
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['contentType'] = "referenceInformation"
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['long_name'] = "crossover reference ground track"
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['description'] = "The RGT number for the crossing data."
            IS2_atl11_gz_attrs[ptx][XT]['rgt']['coordinates'] = \
                "ref_pt delta_time latitude longitude"
            # cycle_number of the crossing track
            IS2_atl11_gz[ptx][XT]['cycle_number'] = mds1[ptx][XT]['cycle_number'].copy()
            IS2_atl11_fill[ptx][XT]['cycle_number'] = attr1[ptx][XT]['cycle_number']['_FillValue']
            IS2_atl11_dims[ptx][XT]['cycle_number'] = None
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number']['units'] = "1"
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number']['long_name'] = "crossover cycle number"
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['cycle_number']['description'] = ("Cycle number for the "
                "crossing data. Number of 91-day periods that have elapsed since ICESat-2 entered "
                "the science orbit. Each of the 1,387 reference ground track (RGTs) is targeted "
                "in the polar regions once every 91 days.")
            # delta time of the crossing track
            IS2_atl11_gz[ptx][XT]['delta_time'] = delta_time['XT'].copy()
            IS2_atl11_fill[ptx][XT]['delta_time'] = delta_time['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['delta_time'] = ['ref_pt']
            IS2_atl11_gz_attrs[ptx][XT]['delta_time'] = {}
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['units'] = "seconds since 2018-01-01"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['long_name'] = "Elapsed GPS seconds"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['standard_name'] = "time"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['calendar'] = "standard"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['description'] = ("Number of GPS "
                "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
                "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
                "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
                "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
                "time in gps_seconds relative to the GPS epoch can be computed.")
            IS2_atl11_gz_attrs[ptx][XT]['delta_time']['coordinates'] = \
                "ref_pt latitude longitude"
            # latitude of the crossover measurement
            IS2_atl11_gz[ptx][XT]['latitude'] = latitude['XT'].copy()
            IS2_atl11_fill[ptx][XT]['latitude'] = latitude['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['latitude'] = ['ref_pt']
            IS2_atl11_gz_attrs[ptx][XT]['latitude'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['units'] = "degrees_north"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['contentType'] = "physicalMeasurement"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['long_name'] = "crossover latitude"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['standard_name'] = "latitude"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['description'] = ("Center latitude of "
                "selected segments")
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['valid_min'] = -90.0
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['valid_max'] = 90.0
            IS2_atl11_gz_attrs[ptx][XT]['latitude']['coordinates'] = \
                "ref_pt delta_time longitude"
            # longitude of the crossover measurement
            IS2_atl11_gz[ptx][XT]['longitude'] = longitude['XT'].copy()
            IS2_atl11_fill[ptx][XT]['longitude'] = longitude['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['longitude'] = ['ref_pt']
            IS2_atl11_gz_attrs[ptx][XT]['longitude'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['units'] = "degrees_east"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['contentType'] = "physicalMeasurement"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['long_name'] = "crossover longitude"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['standard_name'] = "longitude"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['source'] = "ATL06"
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['description'] = ("Center longitude of "
                "selected segments")
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['valid_min'] = -180.0
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['valid_max'] = 180.0
            IS2_atl11_gz_attrs[ptx][XT]['longitude']['coordinates'] = \
                "ref_pt delta_time latitude"
            # computed tide with flexure for the crossover measurement
            IS2_atl11_gz[ptx][XT]['tide_ocean'] = scaled_tide.copy()
            IS2_atl11_fill[ptx][XT]['tide_ocean'] = scaled_tide.fill_value
            IS2_atl11_dims[ptx][XT]['tide_ocean'] = ['ref_pt']
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean'] = collections.OrderedDict()
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['units'] = "meters"
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['contentType'] = "referenceInformation"
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['long_name'] = "Ocean Tide"
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['description'] = ("Ocean Tides with "
                "Near-Grounding Zone Flexure that includes diurnal and semi-diurnal (harmonic analysis), "
                "and longer period tides (dynamic and self-consistent equilibrium).")
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['source'] = tide_source
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['reference'] = \
                "https://doi.org/10.3189/172756410791392790"
            IS2_atl11_gz_attrs[ptx][XT]['tide_ocean']['coordinates'] = \
                "ref_pt delta_time latitude longitude"

    # output flexure correction HDF5 file
    args = (PRD,TIDE_MODEL,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
    file_format = '{0}_{1}_GZ_TIDES_{2}{3}_{4}{5}_{6}_{7}{8}.h5'
    OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(file_format.format(*args))
    # print file information
    logging.info('\t{0}'.format(file_format.format(*args)))
    HDF5_ATL11_corr_write(IS2_atl11_gz, IS2_atl11_gz_attrs,
        FILENAME=OUTPUT_FILE,
        INPUT=GRANULE,
        GROUNDING_ZONE=GROUNDING_ZONE,
        CROSSOVERS=CROSSOVERS,
        FILL_VALUE=IS2_atl11_fill,
        DIMENSIONS=IS2_atl11_dims,
        CLOBBER=True)
    # change the permissions mode
    OUTPUT_FILE.chmod(mode=MODE)

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
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    fileID = h5py.File(FILENAME, clobber)

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
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name
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

# PURPOSE: create arguments parser
def arguments():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Calculates ice sheet grounding zones with ICESat-2
            ATL11 annual land ice height data
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL11 file to run')
    # directory with mask data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=gz.utilities.get_data_path('data'),
        help='Working data directory')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # mean file to remove
    parser.add_argument('--mean-file',
        type=pathlib.Path,
        help='Mean elevation file to remove from the height data')
    # tide model to use
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, default='CATS2008',
        help='Tide model to use in correction')
    # dynamic atmospheric correction
    parser.add_argument('--reanalysis','-R',
        metavar='REANALYSIS', type=str,
        help='Reanalysis model to use in inverse-barometer correction')
    # mean dynamic topography
    parser.add_argument('--sea-level','-S',
        default=False, action='store_true',
        help='Remove mean dynamic topography from heights')
    # run with ATL11 crossovers
    parser.add_argument('--crossovers','-C',
        default=False, action='store_true',
        help='Run ATL11 Crossovers')
    # create test plots
    parser.add_argument('--plot','-P',
        default=False, action='store_true',
        help='Create plots of flexural zone')
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

    # create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # run for each input ATL11 file
    for FILE in args.infile:
        calculate_GZ_ICESat2(args.directory, FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            MEAN_FILE=args.mean_file,
            TIDE_MODEL=args.tide,
            REANALYSIS=args.reanalysis,
            SEA_LEVEL=args.sea_level,
            CROSSOVERS=args.crossovers,
            PLOT=args.plot,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()