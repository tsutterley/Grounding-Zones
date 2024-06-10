#!/usr/bin/env python
u"""
calculate_GZ_ICESat2_ATL03.py
Written by Tyler Sutterley (06/2024)
Calculates ice sheet grounding zones with ICESat-2 data following:
    Brunt et al., Annals of Glaciology, 51(55), 2010
        https://doi.org/10.3189/172756410791392790
    Fricker et al. Geophysical Research Letters, 33(15), 2006
        https://doi.org/10.1029/2006GL026907
    Fricker et al. Antarctic Science, 21(5), 2009
        https://doi.org/10.1017/S095410200999023X

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
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
    io/ATL03.py: reads ICESat-2 ATL03 and ATL09 data files
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 06/2024: attempt to create line string and continue if exception
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        use time functions from timescale.time
    Updated 07/2023: using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 11/2022: verify coordinate reference system attribute from shapefile
    Updated 08/2022: use logging for verbose output of processing run
    Updated 07/2022: place shapely within try/except statement
    Updated 05/2022: use argparse descriptions within documentation
    Updated 03/2021: use utilities to set default path to shapefiles
        replaced numpy bool/int to prevent deprecation warnings
    Updated 01/2021: using argparse to set command line options
        using time module for conversion operations
    Updated 09/2019: using date functions paralleling public repository
    Updated 09/2017: use rcond=-1 in numpy least-squares algorithms
    Written 06/2017
"""
from __future__ import print_function

import re
import logging
import pathlib
import argparse
import operator
import warnings
import itertools
import numpy as np
import grounding_zones as gz

# attempt imports
fiona = gz.utilities.import_dependency('fiona')
h5py = gz.utilities.import_dependency('h5py')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
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

# PURPOSE: compress complete list of valid indices into a set of ranges
def compress_list(i,n):
    for a,b in itertools.groupby(enumerate(i), lambda v: ((v[1]-v[0])//n)*n):
        group = list(map(operator.itemgetter(1),b))
        yield (group[0], group[-1])

# PURPOSE: read ICESat-2 geolocated photon event data (ATL03)
# calculate mean elevation between all dates in file
# calculate inflexion point using elevation surface slopes
# use mean elevation to calculate elevation anomalies
# use anomalies to calculate inward and seaward limits of tidal flexure
def calculate_GZ_ICESat2(base_dir, INPUT_FILE,
    OUTPUT_DIRECTORY=None,
    MODE=0o775):

    # log input file
    logging.info(f'{str(INPUT_FILE)} -->')
    # input granule basename
    GRANULE = INPUT_FILE.name

    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_'
        r'(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$',re.VERBOSE)
    PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = rx.findall(GRANULE).pop()
    # get output directory from input file
    if OUTPUT_DIRECTORY is None:
        OUTPUT_DIRECTORY = INPUT_FILE.parent
    # set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRAN)
    # digital elevation model for each region
    DEM_MODEL = dict(N='ArcticDEM', S='REMA')

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = gz.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = gz.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()

    # read data from input ATL03 file
    IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams = \
        is2tk.io.ATL03.read_main(INPUT_FILE, ATTRIBUTES=True)

    # file format for auxiliary files
    file_format = '{0}_{1}_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
    # grounding zone mask file
    args = (PRD,'GROUNDING_ZONE_MASK',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
    # extract mask values for mask flags to create grounding zone mask
    f1 = OUTPUT_DIRECTORY.joinpath(file_format.format(*args))
    logging.info(str(f1))
    fid1 = h5py.File(f1, mode='r')
    # input digital elevation model file (ArcticDEM or REMA)
    args = (PRD,DEM_MODEL[HEM],YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
    f2 = OUTPUT_DIRECTORY.joinpath(file_format.format(*args))
    logging.info(str(f2))
    fid2 = h5py.File(f2, mode='r')
    # # input sea level for mean dynamic topography
    # args = (PRD,'AVISO_SEA_LEVEL',YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
    # f3 = OUTPUT_DIRECTORY.joinpath(file_format.format(*args))
    # logging.info(str(f3))
    # fid3 = h5py.File(f3, mode='r')

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

    # number of GPS seconds between the GPS epoch
    # and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = IS2_atl03_mds['ancillary_data']['atlas_sdp_gps_epoch']

    # copy variables for outputting to HDF5 file
    IS2_atl03_gz = {}
    IS2_atl03_fill = {}
    IS2_atl03_dims = {}
    IS2_atl03_gz_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl03_gz['ancillary_data'] = {}
    IS2_atl03_gz_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl03_gz['ancillary_data'][key] = IS2_atl03_mds['ancillary_data'][key]
        # Getting attributes of group and included variables
        IS2_atl03_gz_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][key].items():
            IS2_atl03_gz_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        # data and attributes for beam gtx
        val,attrs = is2tk.io.ATL03.read_beam(INPUT_FILE, gtx,
            ATTRIBUTES=True, VERBOSE=False)
        # first photon in the segment (convert to 0-based indexing)
        Segment_Index_begin = val['geolocation']['ph_index_beg'] - 1
        # number of photon events in the segment
        Segment_PE_count = val['geolocation']['segment_ph_cnt']

        # number of photon events
        n_pe, = val['heights']['h_ph'].shape
        # invalid value
        fv = val['geolocation']['sigma_h'].fillvalue

        # check confidence level associated with each photon event
        # -1: Events not associated with a specific surface type
        #  0: noise
        #  1: buffer but algorithm classifies as background
        #  2: low
        #  3: medium
        #  4: high
        # Surface types for signal classification confidence
        # 0=Land; 1=Ocean; 2=SeaIce; 3=LandIce; 4=InlandWater
        ice_sig_conf = np.copy(val['heights']['signal_conf_ph'][:,3])
        ice_sig_low_count = np.count_nonzero(ice_sig_conf > 1)

        # read buffered grounding zone mask
        ice_gz = fid1[gtx]['subsetting']['ice_gz'][:]
        B = fid1[gtx]['subsetting']['ice_gz'].attrs['source']

        # photon event height
        h_ph = np.ma.array(val['heights']['h_ph'], fill_value=fv,
            mask=(ice_sig_conf<=1))
        # digital elevation model elevation
        dem_h = np.ma.array(fid2[gtx]['heights']['dem_h'][:],
            mask=(fid2[gtx]['heights']['dem_h'][:]==fv), fill_value=fv)
        # # mean dynamic topography with invalid values set to 0
        # h_mdt = fid3[gtx]['geophys_corr']['h_mdt'][:]
        # h_mdt[h_mdt == fv] = 0.0

        # ocean tide
        tide = np.zeros_like(val['heights']['h_ph'],dtype=int)
        # dynamic atmospheric correction
        dac = np.zeros_like(val['heights']['h_ph'],dtype=int)
        # geoid height
        geoid = np.zeros_like(val['heights']['h_ph'],dtype=int)
        for j,idx in enumerate(Segment_Index_begin):
            # number of photons in segment
            cnt = Segment_PE_count[j]
            # get ocean tide for each photon event
            tide[idx:idx+cnt] = np.full((cnt),val['geophys_corr']['tide_ocean'][j])
            # get dynamic atmospheric correction for each photon event
            dac[idx:idx+cnt] = np.full((cnt),val['geophys_corr']['dac'][j])
            # get geoid height for each photon event
            geoid[idx:idx+cnt] = np.full((cnt),val['geophys_corr']['geoid'][j])

        # find valid points with GZ for both ATL03 and the interpolated DEM
        valid, = np.nonzero((~h_ph.mask) & (~dem_h.mask) & ice_gz)

        # compress list (separate geosegs into sets of ranges)
        ice_gz_indices = compress_list(valid, 10)
        for imin,imax in ice_gz_indices:
            # find valid indices within range
            i = sorted(set(np.arange(imin,imax+1)) & set(valid))
            # extract lat/lon and convert to polar stereographic
            X,Y = transformer.transform(val['lon_ph'][i],val['lat_ph'][i])
            # shapely LineString object for altimetry segment
            try:
                segment_line = geometry.LineString(np.c_[X, Y])
            except:
                continue
            # determine if line segment intersects previously known GZ
            if segment_line.intersects(mline_obj):
                # horizontal eulerian distance from start of segment
                dist = np.sqrt((X-X[0])**2 + (Y-Y[0])**2)
                # land ice height for grounding zone
                h_gz = h_ph.data[i]
                # mean land ice height from digital elevation model
                h_mean = dem_h[i]

                # deflection from mean height in grounding zone
                dh_gz = h_gz + tide[i] - h_mean # + dac[i]
                # quasi-freeboard: WGS84 elevation - geoid height
                QFB = h_gz - geoid[i]
                # ice thickness from quasi-freeboard and densities
                w_thick = QFB*rho_w/(rho_w-rho_ice)
                # fit with a hard piecewise model to get rough estimate of GZ
                C1,C2,PWMODEL = gz.fit.piecewise_bending(dist, dh_gz, STEP=5)
                # distance from estimated grounding line (0 = grounding line)
                d = (dist - C1[0]).astype(int)
                # determine if spacecraft is approaching coastline
                sco = True if np.mean(h_gz[d<0]) < np.mean(h_gz[d>0]) else False
                # fit physical elastic model
                PGZ,PA,PE,PT,PdH,PEMODEL = gz.fit.elastic_bending(dist, dh_gz,
                    GZ=C1, ORIENTATION=sco, THICKNESS=w_thick, CONF=0.95)
                # linearly interpolate distance to grounding line
                XGZ = np.interp(PGZ[0],dist,X)
                YGZ = np.interp(PGZ[0],dist,X)
                print(XGZ, YGZ, PGZ[0], PGZ[1], PT)

    # close the auxiliary files
    fid1.close()
    fid2.close()
    # fid3.close()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates ice sheet grounding zones with ICESat-2
            ATL03 geolocated photon height data
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL03 file to run')
    # directory with mask data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=gz.utilities.get_data_path('data'),
        help='Working data directory')
    # directory with input/output data
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

    # create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # run for each input ATL03 file
    for FILE in args.infile:
        calculate_GZ_ICESat2(args.directory, FILE,
            OUTPUT_DIRECTORY=args.output_directory,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
