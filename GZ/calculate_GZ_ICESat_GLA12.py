#!/usr/bin/env python
u"""
calculate_GZ_ICESat_GLA12.py
Written by Tyler Sutterley (06/2024)

Calculates ice sheet grounding zones with ICESat data following:
    Brunt et al., Annals of Glaciology, 51(55), 2010
        https://doi.org/10.3189/172756410791392790
    Fricker et al. Geophysical Research Letters, 33(15), 2006
        https://doi.org/10.1029/2006GL026907
    Fricker et al. Antarctic Science, 21(5), 2009
        https://doi.org/10.1017/S095410200999023X

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
        TPXO9-atlas-v5
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
        DAC: AVISO dynamic atmospheric correction (DAC) model
        ERA-Interim: http://apps.ecmwf.int/datasets/data/interim-full-moda
        ERA5: http://apps.ecmwf.int/data-catalogues/era5/?class=ea
        MERRA-2: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
    -S, --sea-level: Remove mean dynamic topography from heights
    -G X, --geoid X: Geoid model to use in correction
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
    fiona: Python wrapper for vector data access functions from the OGR library
        https://fiona.readthedocs.io/en/latest/manual.html
    shapely: PostGIS-ish operations outside a database context for Python
        http://toblerity.org/shapely/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 06/2024: renamed GLAH12 quality summary variable to d_qa_sum
        save output HDF5 files as trajectory type for visualization
    Written 05/2024
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import datetime
import operator
import itertools
import traceback
import collections
import numpy as np
import grounding_zones as gz

# attempt imports
fiona = gz.utilities.import_dependency('fiona')
h5py = gz.utilities.import_dependency('h5py')
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

# PURPOSE: attempt to read the mask variables
def read_grounding_zone_mask(mask_file):
    # check that mask file and variable exists
    for mask in ['d_ice_gz', 'd_mask']:
        try:
            # extract mask values to create grounding zone mask
            fileID = gz.io.multiprocess_h5py(mask_file, mode='r')
            # read buffered grounding zone mask
            ice_gz = fileID['Data_40HZ']['Subsetting'][mask][:].copy()
        except Exception as exc:
            logging.debug(traceback.format_exc())
            pass
        else:
            # close the HDF5 file and return the mask variable
            fileID.close()
            return ice_gz
    # raise value error
    raise KeyError(f'Cannot retrieve mask variable for {str(mask_file)}')

def campaign_bias_correction(campaign: str):
    """
    Additive bias correction based on the laser number
        for a ICESat campaign

    Parameters
    ----------
    campaign: str
        ICESat campaign
    """
    # 
    correction = dict(laser1=0.0, laser2=-0.017, laser3=0.011)
    # find the laser number from the campaign
    laser, = re.findall(r'\d', campaign)
    try:
        return correction[f'laser{laser}']
    except KeyError as exc:
        return 0.0
    
# PURPOSE: compress complete list of valid indices into a set of ranges
def compress_list(i,n):
    for a,b in itertools.groupby(enumerate(i), lambda v: ((v[1]-v[0])//n)*n):
        group = list(map(operator.itemgetter(1),b))
        yield (group[0], group[-1])

# PURPOSE: read ICESat ICESat/GLAS L2 GLA12 Ice Sheet elevation data
# calculate mean elevation between all dates in file
# calculate inflexion point using elevation surface slopes
# use mean elevation to calculate elevation anomalies
# use anomalies to calculate inward and seaward limits of tidal flexure
def calculate_GZ_ICESat(base_dir, INPUT_FILE,
        OUTPUT_DIRECTORY=None,
        HEM=None,
        MEAN_FILE=None,
        TIDE_MODEL=None,
        REANALYSIS=None,
        SEA_LEVEL=False,
        GEOID=None,
        MODE=0o775
    ):

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name
    REGION = dict(N='GL', S='AA')

    # grounded ice line string to determine if segment crosses coastline
    mline_obj, epsg = read_grounded_ice(base_dir, HEM, VARIABLES=[0])
    # projections for converting lat/lon to polar stereographic
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(epsg)
    # transformer object for converting projections
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # densities of seawater and ice
    rho_w = 1030.0
    rho_ice = 917.0

    # compile regular expression operator for extracting information from file
    rx = re.compile((r'GLAH(\d{2})_(\d{3})_(\d{1})(\d{1})(\d{2})_(\d{3})_'
        r'(\d{4})_(\d{1})_(\d{2})_(\d{4})\.H5'), re.VERBOSE)
    # extract parameters from ICESat/GLAS HDF5 file name
    # PRD:  Product number (01, 05, 06, 12, 13, 14, or 15)
    # RL:  Release number for process that created the product = 634
    # RGTP:  Repeat ground-track phase (1=8-day, 2=91-day, 3=transfer orbit)
    # ORB:   Reference orbit number (starts at 1 and increments each time a
    #           new reference orbit ground track file is obtained.)
    # INST:  Instance number (increments every time the satellite enters a
    #           different reference orbit)
    # CYCL:   Cycle of reference orbit for this phase
    # TRK: Track within reference orbit
    # SEG:   Segment of orbit
    # GRAN:  Granule version number
    # TYPE:  File type
    VAR = f'GZ_{REGION[HEM]}'
    try:
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = \
            rx.findall(GRANULE).pop()
    except (ValueError, IndexError):
        # output grounding zone HDF5 file (generic)
        FILENAME = f'{INPUT_FILE.stem}_{VAR}{INPUT_FILE.suffix}'
        FILE_FORMAT = 'generic'
    else:
        # output grounding zone HDF5 file for NSIDC granules
        args = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_{2}_{3}{4}{5}_{6}_{7}_{8}_{9}_{10}.h5'
        FILENAME = file_format.format(*args)
        FILE_FORMAT = 'standard'
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # full path to output file
    OUTPUT_FILE = OUTPUT_DIRECTORY.joinpath(FILENAME)

    # open the HDF5 file for reading
    fid = gz.io.multiprocess_h5py(INPUT_FILE, mode='r')
    # quality summary HDF5 file
    VAR = 'QA'
    if (FILE_FORMAT == 'standard'):
        a1 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f1 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a1))
    elif (FILE_FORMAT == 'generic'):
        file1 = f'{INPUT_FILE.stem}_{VAR}{INPUT_FILE.suffix}'
        f1 = OUTPUT_DIRECTORY.joinpath(file1)
    if not f1.exists():
        return
    # quality summary mask file
    fid1 = gz.io.multiprocess_h5py(f1, mode='r')
    # copy ICESat campaign name from ancillary data
    ancillary_data = fid['ANCILLARY_DATA']
    campaign = ancillary_data.attrs['Campaign'].decode('utf-8')
    # get variables and attributes
    n_40HZ, = fid['Data_40HZ']['Time']['i_rec_ndx'].shape
    rec_ndx_1HZ = fid['Data_1HZ']['Time']['i_rec_ndx'][:].copy()
    rec_ndx_40HZ = fid['Data_40HZ']['Time']['i_rec_ndx'][:].copy()
    # ICESat track number
    i_track_1HZ = fid['Data_1HZ']['Geolocation']['i_track'][:].copy()
    i_track_40HZ = np.zeros((n_40HZ), dtype=i_track_1HZ.dtype)
    # time of ICESat data
    J2000 = fid['Data_40HZ']['DS_UTCTime_40'][:].copy()
    ts = timescale.time.Timescale().from_deltatime(
        J2000, epoch=timescale.time._j2000_epoch,
        standard='UTC')
    # campaign bias correction
    bias_corr = campaign_bias_correction(campaign)
    # saturation correction
    sat_corr = fid['Data_40HZ']['Elevation_Corrections']['d_satElevCorr'][:]
    # Longitude (degrees East)
    d_lon = fid['Data_40HZ']['Geolocation']['d_lon'][:]
    # Latitude (TOPEX/Poseidon ellipsoid degrees North)
    d_lat = fid['Data_40HZ']['Geolocation']['d_lat'][:]
    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
    d_elev = fid['Data_40HZ']['Elevation_Surfaces']['d_elev'][:]
    fv = fid['Data_40HZ']['Elevation_Surfaces']['d_elev'].fillvalue
    # mask for reducing to valid values
    quality_summary = fid1['Data_40HZ']['Quality']['d_qa_sum'][:]
    # get the transform for converting to the latest ITRF
    transform = gz.crs.tp_itrf2008_to_wgs84_itrf2020()
    # transform the data to WGS84 ellipsoid in ITRF2020
    lon, lat, data, tdec = transform.transform(d_lon, d_lat,
        d_elev[:] + sat_corr + bias_corr, ts.year)
    # mask invalid values
    elev = np.ma.array(data, fill_value=fv)
    elev.mask = (d_elev == fv) | np.isnan(elev.data)
    # map 1HZ data to 40HZ data
    for k,record in enumerate(rec_ndx_1HZ):
        # indice mapping the 40HZ data to the 1HZ data
        map_1HZ_40HZ, = np.nonzero(rec_ndx_40HZ == record)
        i_track_40HZ[map_1HZ_40HZ] = i_track_1HZ[k]

    # grounding zone mask
    VAR = 'GROUNDING_ZONE_MASK'
    if (FILE_FORMAT == 'standard'):
        a2 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f2 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a2))
    elif (FILE_FORMAT == 'generic'):
        file2 = f'{INPUT_FILE.stem}_{VAR}_{INPUT_FILE.suffix}'
        f2 = OUTPUT_DIRECTORY.joinpath(file2)
    # extract grounding zone mask
    ice_gz = np.zeros((n_40HZ),dtype=bool)
    ice_gz[:] = read_grounding_zone_mask(f2)

    # mean elevation
    dem_h = np.ma.zeros((n_40HZ), fill_value=fv)
    if MEAN_FILE:
        # read DEM HDF5 file
        fid3 = gz.io.multiprocess_h5py(MEAN_FILE, mode='r')
        var = fid3['Data_40HZ']['Geophysical']['d_DEM_elv']
        dem_h.data[:] = var[:].copy()
        dem_h.mask = (dem_h.data[:] == var.fillvalue)
        fid3.close()
    else:
        # use default DEM within GLA12
        var = fid['Data_40HZ']['Geophysical']['d_DEM_elv']
        dem_h.data[:] = var[:].copy()
        dem_h.mask = (var == var.fillvalue)

    # ocean tide model
    otide = np.ma.zeros((n_40HZ), fill_value=fv)
    if TIDE_MODEL and (FILE_FORMAT == 'standard'):
        VAR = f'{TIDE_MODEL}_TIDES'
        a4 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f4 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a4))
    elif TIDE_MODEL and (FILE_FORMAT == 'generic'):
        VAR = f'{TIDE_MODEL}_TIDES'
        file4 = f'{INPUT_FILE.stem}_{VAR}_{INPUT_FILE.suffix}'
        f4 = OUTPUT_DIRECTORY.joinpath(file4)
    if TIDE_MODEL:
        fid4 = gz.io.multiprocess_h5py(f4, mode='r')
        var = fid4['Data_40HZ']['Geophysical']['d_ocElv']
        otide.data[:] = var[:]
        otide.mask = (otide.data[:] == var.fillvalue)
        fid4.close()
    else:
        # use default tide model
        var = fid['Data_40HZ']['Geophysical']['d_ocElv']
        otide.data[:] = var[:]
        otide.mask = (otide.data[:] == var.fillvalue)

    # inverse barometer or dynamic atmosphere correction
    IB = np.ma.zeros((n_40HZ), fill_value=fv)
    IB.mask = np.zeros((n_40HZ), dtype=bool)
    if REANALYSIS and (FILE_FORMAT == 'standard'):
        VAR = 'DAC' if (REANALYSIS == 'DAC') else f'{REANALYSIS}_IB'
        a5 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f5 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a5))
    elif REANALYSIS and (FILE_FORMAT == 'generic'):
        VAR = 'DAC' if (REANALYSIS == 'DAC') else f'{REANALYSIS}_IB'
        file5 = f'{INPUT_FILE.stem}_{VAR}_{INPUT_FILE.suffix}'
        f5 = OUTPUT_DIRECTORY.joinpath(file5)
    if REANALYSIS:
        fid5 = gz.io.multiprocess_h5py(f5, mode='r')
        key = 'd_dacElv' if (REANALYSIS == 'DAC') else 'd_ibElv'
        var = fid5['Data_40HZ']['Geophysical'][key]
        IB.data[:] = var[:]
        IB.mask = (IB.data[:] == var.fillvalue)
        fid5.close()

    # mean dynamic topography
    mdt = np.ma.zeros((n_40HZ),fill_value=fv)
    mdt.mask = np.zeros((n_40HZ),dtype=bool)
    if SEA_LEVEL and (FILE_FORMAT == 'standard'):
        VAR = 'DAC' if (REANALYSIS == 'DAC') else f'{REANALYSIS}_IB'
        a6 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f6 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a6))
    elif SEA_LEVEL and (FILE_FORMAT == 'generic'):
        VAR = 'DAC' if (REANALYSIS == 'DAC') else f'{REANALYSIS}_IB'
        file6 = f'{INPUT_FILE.stem}_{VAR}_{INPUT_FILE.suffix}'
        f6 = OUTPUT_DIRECTORY.joinpath(file6)
    if SEA_LEVEL:
        fid6 = gz.io.multiprocess_h5py(f6, mode='r')
        var = fid6['Data_40HZ']['Geophysical']['d_MDT_elv']
        mdt.data[:] = var[:]
        mdt.mask = (mdt.data[:] == var.fillvalue)
        fid6.close()

    # geoid height
    gdHt = np.ma.zeros((n_40HZ), fill_value=fv)
    if GEOID and (FILE_FORMAT == 'standard'):
        VAR = f'{GEOID}_GEOID'
        a7 = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        f7 = OUTPUT_DIRECTORY.joinpath(file_format.format(*a7))
    elif GEOID and (FILE_FORMAT == 'generic'):
        VAR = f'{GEOID}_GEOID'
        file7 = f'{INPUT_FILE.stem}_{VAR}_{INPUT_FILE.suffix}'
        f7 = OUTPUT_DIRECTORY.joinpath(file7)
    if GEOID:
        fid7 = gz.io.multiprocess_h5py(f7, mode='r')
        var = fid7['Data_40HZ']['Geophysical']['d_gdHt']
        gdHt.data[:] = var[:]
        gdHt.mask = (gdHt.data[:] == var.fillvalue)
        fid7.close()
    else:
        # use default geoid height
        var = fid['Data_40HZ']['Geophysical']['d_gdHt']
        gdHt.data[:] = var[:]
        gdHt.mask = (gdHt.data[:] == var.fillvalue)

    # flag that a valid grounding zone fit has been found
    valid_fit = False
    # outputs of grounding zone fit
    Data_GZ = collections.OrderedDict()
    Data_GZ['DS_UTCTime_40'] = []
    Data_GZ['i_rec_ndx'] = []
    Data_GZ['i_track'] = []
    Data_GZ['d_lat'] = []
    Data_GZ['d_lon'] = []
    # Data_GZ['d_ocElv'] = []
    Data_GZ['d_gz_sigma'] = []
    Data_GZ['d_e_mod'] = []
    Data_GZ['d_e_mod_sigma'] = []
    # Data_GZ['d_H_ice'] = []
    # Data_GZ['d_delta_h'] = []

    # for each repeat ground track
    for rgt in np.unique(i_track_40HZ):
        # find valid points for track with GZ
        valid_mask = np.logical_not(elev.mask | quality_summary) 
        valid, = np.nonzero((i_track_40HZ == rgt) & valid_mask & ice_gz)

        # compress list (separate values into sets of ranges)
        ice_gz_indices = compress_list(valid,10)
        for imin,imax in ice_gz_indices:
            # find valid indices within range
            i = sorted(set(np.arange(imin,imax+1)) & set(valid))
            # extract lat/lon and convert to polar stereographic
            X,Y = transformer.transform(lon[i], lat[i])
            # shapely LineString object for altimetry segment
            try:
                segment_line = geometry.LineString(np.c_[X, Y])
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
                    iint = np.argmin((Y-yi)**2 + (X-xi)**2)
                # horizontal eulerian distance from start of segment
                dist = np.sqrt((X-X[0])**2 + (Y-Y[0])**2)
                # land ice height for grounding zone
                h_gz = elev[i]
                # mean land ice height from digital elevation model
                h_mean = dem_h.data[i]
                # geoid height
                h_geoid = gdHt[i]

                # ocean tide height for scaling model
                h_tide = np.ma.array(otide.data[i], fill_value=fv)
                h_tide.mask = otide.mask[i]
                # inverse-barometer response
                h_ib = np.ma.array(IB.data[i], fill_value=fv)
                h_ib.mask = IB.mask[i]

                # deflection from mean land ice height in grounding zone
                dh_gz = h_gz - h_mean
                # quasi-freeboard: WGS84 elevation - geoid height
                QFB = h_gz - (h_geoid + mdt[i])
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
                            CONF=0.95, XOUT=i)
                    except Exception as exc:
                        logging.debug(traceback.format_exc())
                        pass
                    # copy grounding zone parameters to get best fit
                    if (GZ[1] < PGZ[1]) & (GZ[1] != 0.0):
                        PGZ = np.copy(GZ)
                        model_scale = np.copy(PA[0])
                        PEMODEL = np.copy(MODEL)
                    # use parameters if fit significance is within tolerance
                    if (GZ[1] < 400.0):
                        break
                # skip saving parameters if no valid solution was found
                if np.logical_not(np.isfinite(PGZ[0])):
                    continue
                # set valid beam fit flag
                valid_fit = True

                # linearly interpolate distance to grounding line
                GZrec = np.interp(PGZ[0],dist,rec_ndx_40HZ[i])
                GZlat = np.interp(PGZ[0],dist,lat[i])
                GZlon = np.interp(PGZ[0],dist,lon[i])
                GZtime = np.interp(PGZ[0],dist,J2000[i])
                # append outputs of grounding zone fit
                # save all outputs (not just within tolerance)
                Data_GZ['DS_UTCTime_40'].append(GZtime)
                Data_GZ['i_rec_ndx'].append(GZrec)
                Data_GZ['i_track'].append(rgt)
                Data_GZ['d_lat'].append(GZlat)
                Data_GZ['d_lon'].append(GZlon)
                # Data_GZ['d_ocElv'].append(PA)
                Data_GZ['d_gz_sigma'].append(PGZ[1])
                Data_GZ['d_e_mod'].append(PE[0]/1e9)
                Data_GZ['d_e_mod_sigma'].append(PE[1]/1e9)
                # Data_GZ['d_H_ice'].append(PT)
                # Data_GZ['d_delta_h'].append(PdH)

    # if no valid grounding zone fit has been found
    # skip saving variables and attributes
    if not valid_fit:
        return

    # copy variables for outputting to HDF5 file
    IS_gla12_gz = dict(Data_GZ={})
    IS_gla12_gz_attrs = dict(Data_GZ={})
    # copy variables as numpy arrays
    IS_gla12_gz['Data_GZ'] = {key:np.array(val) for key, val in Data_GZ.items()}

    # copy global file attributes of interest
    global_attribute_list = ['title','comment','summary','license',
        'references','AccessConstraints','CitationforExternalPublication',
        'contributor_role','contributor_name','creator_name','creator_email',
        'publisher_name','publisher_email','publisher_url','platform','instrument',
        'processing_level','date_created','spatial_coverage_type','history',
        'keywords','keywords_vocabulary','naming_authority','project','time_type',
        'date_type','time_coverage_start','time_coverage_end',
        'time_coverage_duration','source','HDFVersion','identifier_product_type',
        'identifier_product_format_version','Conventions','institution',
        'ReprocessingPlanned','ReprocessingActual','LocalGranuleID',
        'ProductionDateTime','LocalVersionID','PGEVersion','OrbitNumber',
        'StartOrbitNumber','StopOrbitNumber','EquatorCrossingLongitude',
        'EquatorCrossingTime','EquatorCrossingDate','ShortName','VersionID',
        'InputPointer','RangeBeginningTime','RangeEndingTime','RangeBeginningDate',
        'RangeEndingDate','PercentGroundHit','OrbitQuality','Cycle','Track',
        'Instrument_State','Timing_Bias','ReferenceOrbit','SP_ICE_PATH_NO',
        'SP_ICE_GLAS_StartBlock','SP_ICE_GLAS_EndBlock','Instance','Range_Bias',
        'Instrument_State_Date','Instrument_State_Time','Range_Bias_Date',
        'Range_Bias_Time','Timing_Bias_Date','Timing_Bias_Time',
        'identifier_product_doi','identifier_file_uuid',
        'identifier_product_doi_authority']
    for att in global_attribute_list:
        IS_gla12_gz_attrs[att] = fid.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_gz_attrs['Campaign'] = campaign
    # save HDF5 as trajectory type
    IS_gla12_gz_attrs['featureType'] = 'trajectory'

    # add attributes for input GLA12 file
    IS_gla12_gz_attrs['lineage'] = GRANULE
    # update geospatial ranges for ellipsoid
    IS_gla12_gz_attrs['geospatial_lat_min'] = np.min(lat)
    IS_gla12_gz_attrs['geospatial_lat_max'] = np.max(lat)
    IS_gla12_gz_attrs['geospatial_lon_min'] = np.min(lon)
    IS_gla12_gz_attrs['geospatial_lon_max'] = np.max(lon)
    IS_gla12_gz_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_gz_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_gz_attrs['geospatial_ellipsoid'] = "WGS84"

    # group attributes for grounding zone variables
    IS_gla12_gz_attrs['Data_GZ']['Description'] = ("The Data_GZ "
        "group contains statistic data at grounding zone locations.")

    # J2000 time
    IS_gla12_gz_attrs['Data_GZ']['DS_UTCTime_40'] = collections.OrderedDict()
    IS_gla12_gz_attrs['Data_GZ']['DS_UTCTime_40']['units'] = \
        "seconds since 2000-01-01T12:00:00"
    IS_gla12_gz_attrs['Data_GZ']['DS_UTCTime_40']['long_name'] = \
        "Transmit time in J2000 seconds"
    IS_gla12_gz_attrs['Data_GZ']['DS_UTCTime_40']['standard_name'] = "time"
    IS_gla12_gz_attrs['Data_GZ']['DS_UTCTime_40']['calendar'] = "standard"
    IS_gla12_gz_attrs['Data_GZ']['DS_UTCTime_40']['description'] = \
        ("Number of UTC seconds since the J2000 epoch "
        "(2000-01-01T12:00:00.000000Z)")
    IS_gla12_gz_attrs['Data_GZ']['DS_UTCTime_40']['coordinates'] = \
        "d_lat d_lon"

    # record
    IS_gla12_gz_attrs['Data_GZ']['i_rec_ndx'] = collections.OrderedDict()
    IS_gla12_gz_attrs['Data_GZ']['i_rec_ndx']['units'] = "1"
    IS_gla12_gz_attrs['Data_GZ']['i_rec_ndx']['long_name'] = \
        "GLAS Record Index"
    IS_gla12_gz_attrs['Data_GZ']['i_rec_ndx']['description'] = \
        ("Unique index that relates this record to the corresponding "
        "record(s) in each GLAS data product")
    IS_gla12_gz_attrs['Data_GZ']['i_rec_ndx']['coordinates'] = \
        "d_lat d_lon"

    # ground track
    IS_gla12_gz_attrs['Data_GZ']['i_track'] = collections.OrderedDict()
    IS_gla12_gz_attrs['Data_GZ']['i_track']['units'] = "1"
    IS_gla12_gz_attrs['Data_GZ']['i_track']['long_name'] = "Track"
    IS_gla12_gz_attrs['Data_GZ']['i_track']['description'] = \
        "GLAS ground track number"
    IS_gla12_gz_attrs['Data_GZ']['i_track']['coordinates'] = \
        "d_lat d_lon"
    
    # latitude
    IS_gla12_gz_attrs['Data_GZ']['d_lat'] = collections.OrderedDict()
    IS_gla12_gz_attrs['Data_GZ']['d_lat']['units'] = "degrees_north"
    IS_gla12_gz_attrs['Data_GZ']['d_lat']['contentType'] = \
        "physicalMeasurement"
    IS_gla12_gz_attrs['Data_GZ']['d_lat']['long_name'] = "Latitude"
    IS_gla12_gz_attrs['Data_GZ']['d_lat']['standard_name'] = "latitude"
    IS_gla12_gz_attrs['Data_GZ']['d_lat']['description'] = \
        "Latitude of estimated grounding zone location"
    IS_gla12_gz_attrs['Data_GZ']['d_lat']['valid_min'] = -90.0
    IS_gla12_gz_attrs['Data_GZ']['d_lat']['valid_max'] = 90.0

    # longitude
    IS_gla12_gz_attrs['Data_GZ']['d_lon'] = collections.OrderedDict()
    IS_gla12_gz_attrs['Data_GZ']['d_lon']['units'] = "degrees_east"
    IS_gla12_gz_attrs['Data_GZ']['d_lon']['contentType'] = \
        "physicalMeasurement"
    IS_gla12_gz_attrs['Data_GZ']['d_lon']['long_name'] = "Longitude"
    IS_gla12_gz_attrs['Data_GZ']['d_lon']['standard_name'] = "longitude"
    IS_gla12_gz_attrs['Data_GZ']['d_lon']['description'] = \
        "Longitude of estimated grounding zone location"
    IS_gla12_gz_attrs['Data_GZ']['d_lon']['valid_min'] = -180.0
    IS_gla12_gz_attrs['Data_GZ']['d_lon']['valid_max'] = 180.0

    # uncertainty of the grounding zone
    IS_gla12_gz_attrs['Data_GZ']['d_gz_sigma'] = collections.OrderedDict()
    IS_gla12_gz_attrs['Data_GZ']['d_gz_sigma']['units'] = "meters"
    IS_gla12_gz_attrs['Data_GZ']['d_gz_sigma']['contentType'] = \
        "physicalMeasurement"
    IS_gla12_gz_attrs['Data_GZ']['d_gz_sigma']['long_name'] = \
        "grounding zone uncertainty"
    IS_gla12_gz_attrs['Data_GZ']['d_gz_sigma']['source'] = "GLA12"
    IS_gla12_gz_attrs['Data_GZ']['d_gz_sigma']['description'] = \
        ("Uncertainty in grounding zone location derived by the physical "
        "elastic bending model")
    IS_gla12_gz_attrs['Data_GZ']['d_gz_sigma']['coordinates'] = \
        "d_lat d_lon"

    # effective elastic modulus
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod'] = collections.OrderedDict()
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod']['units'] = "GPa"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod']['contentType'] = \
        "physicalMeasurement"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod']['long_name'] = \
        "Elastic modulus"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod']['source'] = "GLA12"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod']['description'] = \
        ("Effective Elastic modulus of ice estimating using an "
        "elastic beam model")
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod']['coordinates'] = \
        "d_lat d_lon"

    # uncertainty of the elastic modulus
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod_sigma'] = collections.OrderedDict()
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod_sigma']['units'] = "GPa"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod_sigma']['contentType'] = \
        "physicalMeasurement"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod_sigma']['long_name'] = \
        "Elastic modulus uncertainty"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod_sigma']['source'] = "GLA12"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod_sigma']['description'] = \
        "Uncertainty in the effective Elastic modulus of ice"
    IS_gla12_gz_attrs['Data_GZ']['d_e_mod_sigma']['coordinates'] = \
        "d_lat d_lon"
    
    # output grounding zone values to HDF5
    HDF5_GLA12_corr_write(IS_gla12_gz, IS_gla12_gz_attrs,
        FILENAME=OUTPUT_FILE, CLOBBER=True)
    # change the permissions of the output file
    OUTPUT_FILE.chmod(MODE)

# PURPOSE: outputting the grounding zone values for ICESat data to HDF5
def HDF5_GLA12_corr_write(IS_gla12_gz, IS_gla12_attrs,
    FILENAME='', CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    fileID = h5py.File(FILENAME, clobber)
    # create 40HZ HDF5 records
    h5 = dict(Data_GZ={})

    # add HDF5 file attributes
    for att_name,att_val in IS_gla12_attrs.items():
        # skip if attribute is a dictionary
        if not isinstance(att_val, dict):
            fileID.attrs[att_name] = att_val

    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version

    # create grounding zone group
    fileID.create_group('Data_GZ')
    group = fileID['Data_GZ']
    # add HDF5 grounding zone group attributes
    for att_name,att_val in IS_gla12_attrs['Data_GZ'].items():
        # skip if attribute is a dictionary
        if not isinstance(att_val, dict):
            fileID.attrs[att_name] = att_val

    # for each variable in the group
    for key,val in IS_gla12_gz['Data_GZ'].items():
        # Defining the HDF5 dataset variables
        h5['Data_GZ'][key] = group.create_dataset(key,
            np.shape(val), data=val, dtype=val.dtype,
            compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS_gla12_attrs['Data_GZ'][key].items():
            # skip if attribute is a dictionary
            if not isinstance(att_val, dict):
                h5['Data_GZ'][key].attrs[att_name] = att_val
        # create or attach dimensions
        if (key == 'DS_UTCTime_40'):
            h5['Data_GZ'][key].make_scale('DS_UTCTime_40')
        else:
            dim = h5['Data_GZ']['DS_UTCTime_40']
            h5['Data_GZ'][key].dims[0].attach_scale(dim)

    # Closing the HDF5 file
    fileID.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates ice sheet grounding zones with
            ICESat/GLAS L2 GLA12 Antarctic and Greenland Ice Sheet
            elevation data
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat files to run')
    # directory with mask data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=gz.utilities.get_data_path('data'),
        help='Working data directory')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # region of interest to run
    parser.add_argument('--hemisphere','-H',
        type=str, default='S', choices=('N','S'),
        help='Region of interest to run')
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
    # geoid height
    parser.add_argument('--geoid','-G',
        metavar='GEOID', type=str,
        help='Geoid height model to use in correction')
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

    # run for each ICESat file
    for FILE in args.infile:
        # run the program with the specified arguments
        calculate_GZ_ICESat(args.directory, FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            HEM=args.hemisphere,
            MEAN_FILE=args.mean_file,
            TIDE_MODEL=args.tide,
            REANALYSIS=args.reanalysis,
            SEA_LEVEL=args.sea_level,
            GEOID=args.geoid,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()