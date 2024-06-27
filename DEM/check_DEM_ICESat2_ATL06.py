#!/usr/bin/env python
u"""
check_DEM_ICESat2_ATL06.py
Written by Tyler Sutterley (05/2024)
Determines which digital elevation model tiles to read for a given ATL06 file

ArcticDEM 2m digital elevation model tiles
    http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/
    http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/

REMA 8m digital elevation model tiles
    http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/
    http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/

GIMP 30m digital elevation model tiles computed with nsidc_convert_GIMP_DEM.py
    https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0645.001/

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    --model X: Set the digital elevation model (REMA, ArcticDEM, GIMP) to run
    -V, --verbose: Output information about each input file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    fiona: Python wrapper for vector data access functions from the OGR library
        https://fiona.readthedocs.io/en/latest/manual.html
    shapely: PostGIS-ish operations outside a database context for Python
        http://toblerity.org/shapely/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    io/ATL06.py: reads ICESat-2 land ice along-track height data files

REFERENCES:
    https://www.pgc.umn.edu/guides/arcticdem/data-description/
    https://www.pgc.umn.edu/guides/rema/data-description/
    https://nsidc.org/data/nsidc-0645/versions/1

UPDATE HISTORY:
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 08/2023: create s3 filesystem when using s3 urls as input
    Updated 07/2023: using pathlib to define and operate on paths
        use geoms attribute for shapely 2.0 compliance
    Updated 12/2022: single implicit import of grounding zone tools
        refactored ICESat-2 data product read programs under io
    Updated 11/2022: new ArcticDEM and REMA mosaic index shapefiles
        verify coordinate reference system attribute from shapefile
    Updated 05/2022: use argparse descriptions within documentation
    Updated 01/2021: using argparse to set command line options
        using standalone ATL06 reader to get geolocations
    Written 09/2019
"""
from __future__ import print_function

import re
import logging
import pathlib
import argparse
import numpy as np
import grounding_zones as gz

# attempt imports
fiona = gz.utilities.import_dependency('fiona')
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
pyproj = gz.utilities.import_dependency('pyproj')
shapely = gz.utilities.import_dependency('shapely')
shapely.geometry = gz.utilities.import_dependency('shapely.geometry')

# digital elevation models
elevation_dir = {}
elevation_tile_index = {}
# ArcticDEM
elevation_dir['ArcticDEM'] = ['ArcticDEM']
elevation_tile_index['ArcticDEM'] = 'ArcticDEM_Mosaic_Index_v3_shp.zip'
# GIMP DEM
elevation_dir['GIMP'] = ['GIMP','30m']
elevation_tile_index['GIMP'] = 'gimpdem_Tile_Index_Rel1.1.zip'
# REMA DEM
elevation_dir['REMA'] = ['REMA']
elevation_tile_index['REMA'] = 'REMA_Mosaic_Index_v2_shp.zip'

# PURPOSE: set the DEM model to interpolate based on the input granule
def set_DEM_model(GRANULE):
    if GRANULE in ('10','11','12'):
        DEM_MODEL = 'REMA'
    elif GRANULE in ('02','03','04','05','06'):
        DEM_MODEL = 'ArcticDEM'
    return DEM_MODEL

# PURPOSE: read zip file containing index shapefiles for finding DEM tiles
def read_DEM_index(index_file, DEM_MODEL):
    # read the compressed shape file and extract entities
    index_file = pathlib.Path(index_file).expanduser().absolute()
    shape = fiona.open(f'zip://{str(index_file)}')
    # extract coordinate reference system
    if ('init' in shape.crs.keys()):
        epsg = pyproj.CRS(shape.crs['init']).to_epsg()
    else:
        epsg = pyproj.CRS(shape.crs).to_epsg()
    # extract attribute indice for DEM tile (REMA,GIMP) or name (ArcticDEM)
    if (DEM_MODEL == 'REMA'):
        # REMA index file attributes:
        # name: DEM mosaic name for tile (file name without suffix)
        # tile: DEM tile identifier (IMy_IMx)
        # nd_value: fill value for elements with no data
        # resolution: DEM horizontal spatial resolution (meters)
        # creationda: creation date
        # raster: (empty)
        # fileurl: link to file on PGC server
        # spec_type: specific type (DEM)
        # qual: density of scenes within tile (0 to 1)
        # reg_src: DEM registration source (ICESat or neighbor align)
        # num_gcps: number of ground control points
        # meanresz: mean vertical residual (meters)
        # active: (1)
        # qc: (2)
        # rel_ver: release version
        # num_comp: number of components
        # st_area_sh: tile area (meters^2)
        # st_length_: perimeter length of tile (meters)
        field = 'tile'
    elif (DEM_MODEL == 'GIMP'):
        # GIMP index file attributes (from make_GIMP_tile_shapefile.py):
        # name: DEM mosaic name for tile (file name without suffix)
        # tile: DEM tile identifier (IMy_IMx)
        # nd_value: fill value for elements with no data
        # resolution: DEM horizontal spatial resolution (meters)
        # fileurl: link to file on NSIDC server
        # spec_type: specific type (DEM)
        # reg_src: DEM registration source (ICESat or neighbor align)
        # rel_ver: release version
        # num_comp: number of components
        # st_area_sh: tile area (meters^2)
        # st_length_: perimeter length of tile (meters)
        field = 'tile'
    elif (DEM_MODEL == 'ArcticDEM'):
        # ArcticDEM index file attributes:
        # objectid: DEM tile object identifier for sub-tile
        # name: DEM mosaic name for sub-tile (file name without suffix)
        # tile: DEM tile identifier (IMy_IMx) (non-unique for sub-tiles)
        # nd_value: fill value for elements with no data
        # resolution: DEM horizontal spatial resolution (meters)
        # creationda: creation date
        # raster: (empty)
        # fileurl: link to file on PGC server
        # spec_type: specific type (DEM)
        # qual: density of scenes within tile (0 to 1)
        # reg_src: DEM registration source (ICESat or neighbor align)
        # num_gcps: number of ground control points
        # meanresz: mean vertical residual (meters)
        # active: (1)
        # qc: (2)
        # rel_ver: release version
        # num_comp: number of components
        # st_area_sh: tile area (meters^2)
        # st_length_: perimeter length of tile (meters)
        field = 'name'
    # create python dictionary for each polygon object
    poly_dict = {}
    attrs_dict = {}
    # extract the entities and assign by tile name
    for i,ent in enumerate(shape.values()):
        # tile or name attributes
        if DEM_MODEL in ('REMA','GIMP'):
            tile = str(ent['properties'][field])
        else:
            tile, = re.findall(r'^(\d+_\d+_\d+_\d+)',ent['properties'][field])
        # extract attributes and assign by tile
        attrs_dict[tile] = {}
        for key,val in ent['properties'].items():
            attrs_dict[tile][key] = val
        # upper-left, upper-right, lower-right, lower-left, upper-left
        ul,ur,lr,ll,ul2 = ent['geometry']['coordinates'].pop()
        # tile boundaries
        attrs_dict[tile]['xmin'] = ul[0]
        attrs_dict[tile]['xmax'] = lr[0]
        attrs_dict[tile]['ymin'] = lr[1]
        attrs_dict[tile]['ymax'] = ul[1]
        # extract Polar Stereographic coordinates for entity
        x = [ul[0],ur[0],lr[0],ll[0],ul2[0]]
        y = [ul[1],ur[1],lr[1],ll[1],ul2[1]]
        poly_obj = shapely.geometry.Polygon(np.c_[x, y])
        # Valid Polygon may not possess overlapping exterior or interior rings
        if (not poly_obj.is_valid):
            poly_obj = poly_obj.buffer(0)
        poly_dict[tile] = poly_obj
    # close the file
    shape.close()
    # return the dictionaries of polygon objects and attributes
    return (poly_dict,attrs_dict,epsg)

# PURPOSE: read ICESat-2 data and determine which DEM tiles to read
def check_DEM_ICESat2_ATL06(INPUT_FILE,
    DIRECTORY=None,
    DEM_MODEL=None):

    # log input file
    GRANULE = INPUT_FILE.name
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = \
        rx.findall(GRANULE).pop()

    # check if data is an s3 presigned url
    if str(INPUT_FILE).startswith('s3:'):
        client = is2tk.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = is2tk.utilities.s3_filesystem()
        INPUT_FILE = session.open(INPUT_FILE, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(INPUT_FILE).expanduser().absolute()

    # read data from input ATL06 file
    IS2_atl06_mds,IS2_atl06_attrs,IS2_atl06_beams = \
        is2tk.io.ATL06.read_granule(INPUT_FILE, ATTRIBUTES=True)

    # set the  digital elevation model based on ICESat-2 granule
    DEM_MODEL = set_DEM_model(GRAN) if (DEM_MODEL is None) else DEM_MODEL
    # regular expression pattern for extracting parameters from ArcticDEM name
    rx1 = re.compile(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+m)_(.*?)$', re.VERBOSE)
    # directory setup
    DIRECTORY = pathlib.Path(DIRECTORY).expanduser().absolute()
    # full path to DEM directory
    elevation_directory = DIRECTORY.joinpath(*elevation_dir[DEM_MODEL])
    # zip file containing index shapefiles for finding DEM tiles
    index_file = elevation_directory.joinpath(elevation_tile_index[DEM_MODEL])
    # read index file for determining which tiles to read
    tile_dict, tile_attrs, tile_epsg = read_DEM_index(index_file, DEM_MODEL)

    # pyproj transformer for converting from latitude/longitude
    # into DEM tile coordinates
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(tile_epsg)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # list of all tiles that are not presently in the file system
    all_tiles = []
    # for each input beam within the file
    for gtx in sorted(IS2_atl06_beams):
        # number of segments
        val = IS2_atl06_mds[gtx]['land_ice_segments']
        n_seg = len(val['segment_id'])
        # invalid value
        fv = IS2_atl06_attrs[gtx]['land_ice_segments']['h_li']['_FillValue']

        # extract lat/lon and set masks
        latitude = np.ma.array(val['latitude'], fill_value=fv)
        latitude.mask = (val['latitude'] == fv)
        longitude = np.ma.array(val['longitude'], fill_value=fv)
        longitude.mask = (val['longitude'] == fv)
        # convert projection from latitude/longitude to tile EPSG
        X,Y = transformer.transform(longitude, latitude)
        # convert reduced x and y to shapely multipoint object
        xy_point = shapely.geometry.MultiPoint(np.c_[X, Y])

        # create complete masks for each DEM tile
        intersection_map = {}
        for key,poly_obj in tile_dict.items():
            # create empty intersection map array
            intersection_map[key] = np.zeros((n_seg),dtype=np.int64)
            # finds if points are encapsulated (within tile)
            int_test = poly_obj.intersects(xy_point)
            if int_test:
                # extract intersected points
                int_map = list(map(poly_obj.intersects, xy_point.geoms))
                int_indices, = np.nonzero(int_map)
                # set distributed_map indices to True for intersected points
                intersection_map[key][int_indices] = True
        # find valid tiles and free up memory from invalid tiles
        valid_tiles = [k for k,v in intersection_map.items() if v.any()]
        invalid_tiles = sorted(set(intersection_map.keys()) - set(valid_tiles))
        for key in invalid_tiles:
            intersection_map[key] = None

        # for each valid tile
        for key in valid_tiles:
            sub = tile_attrs[key]["tile"]
            name = tile_attrs[key]["name"]
            # read central DEM file (geotiff within gzipped tar file)
            tar = f'{name}.tar.gz'
            elevation_file = elevation_directory.joinpath(sub, tar)
            if not elevation_file.exists():
                all_tiles.append(sub)
            # buffer using neighbor tiles (REMA/GIMP) or sub-tiles (ArcticDEM)
            if (DEM_MODEL == 'REMA'):
                # REMA tiles to read to buffer the image
                IMy,IMx = np.array(re.findall(r'(\d+)_(\d+)',sub).pop(),dtype='i')
                # neighboring tiles for buffering DEM (LB,LM,LT,CB,CT,RB,RM,RT)
                xtiles = [IMx-1,IMx-1,IMx-1,IMx,IMx,IMx+1,IMx+1,IMx+1] # LLLCCRRR
                ytiles = [IMy-1,IMy,IMy+1,IMy-1,IMy+1,IMy-1,IMy,IMy+1] # BMTBTBMT
                for xtl,ytl in zip(xtiles,ytiles):
                    # read DEM file (geotiff within gzipped tar file)
                    bkey = f'{ytl:02d}_{xtl:02d}'
                    # if buffer file is a valid tile within the DEM
                    # if file doesn't exist: will be all fill value with all mask
                    if bkey in tile_attrs.keys():
                        bsub = tile_attrs[bkey]["tile"]
                        btar = f'{tile_attrs[bkey]["name"]}.tar.gz'
                        buffer_file = elevation_directory.joinpath(bkey, btar)
                        if not buffer_file.exists():
                            all_tiles.append(bsub)
            elif (DEM_MODEL == 'GIMP'):
                # GIMP tiles to read to buffer the image
                IMx,IMy = np.array(re.findall(r'(\d+)_(\d+)',sub).pop(),dtype='i')
                # neighboring tiles for buffering DEM (LB,LM,LT,CB,CT,RB,RM,RT)
                xtiles = [IMx-1,IMx-1,IMx-1,IMx,IMx,IMx+1,IMx+1,IMx+1] # LLLCCRRR
                ytiles = [IMy-1,IMy,IMy+1,IMy-1,IMy+1,IMy-1,IMy,IMy+1] # BMTBTBMT
                for xtl,ytl in zip(xtiles,ytiles):
                    # read DEM file (geotiff within gzipped tar file)
                    bkey = f'{xtl:d}_{ytl:d}'
                    # if buffer file is a valid tile within the DEM
                    # if file doesn't exist: will be all fill value with all mask
                    if bkey in tile_attrs.keys():
                        bsub = tile_attrs[bkey]["tile"]
                        btar = f'{tile_attrs[bkey]["name"]}.tar.gz'
                        buffer_file = elevation_directory.joinpath(bkey, btar)
                        if not buffer_file.exists():
                            all_tiles.append(bsub)
            elif (DEM_MODEL == 'ArcticDEM'):
                # ArcticDEM sub-tiles to read to buffer the image
                # extract parameters from tile filename
                IMy,IMx,STx,STy,res,vers = rx1.findall(name).pop()
                IMy,IMx,STx,STy = np.array([IMy,IMx,STx,STy],dtype='i')
                # neighboring tiles for buffering DEM (LB,LM,LT,CB,CT,RB,RM,RT)
                # LLLCCRRR
                xtiles = [IMx+(STx-2)//2,IMx+(STx-2)//2,IMx+(STx-2)//2,IMx,IMx,
                    IMx+STx//2,IMx+STx//2,IMx+STx//2]
                xsubtiles = [(STx-2) % 2 + 1,(STx-2) % 2 + 1,(STx-2) % 2 + 1,
                    STx,STx,STx % 2 + 1,STx % 2 + 1,STx % 2 + 1]
                # BMTBTBMT
                ytiles = [IMy+(STy-2)//2,IMy,IMy+STy//2,IMy+(STy-2)//2,
                    IMy+STy//2,IMy+(STy-2)//2,IMy,IMy+STy//2]
                ysubtiles = [(STy-2) % 2 + 1,STy,STy % 2 + 1,(STy-2) % 2 + 1,
                    STy % 2 + 1,(STy-2) % 2 + 1,STy,STy % 2 + 1]
                # for each buffer tile and sub-tile
                for xtl,ytl,xs,ys in zip(xtiles,ytiles,xsubtiles,ysubtiles):
                    # read DEM file (geotiff within gzipped tar file)
                    bkey = f'{ytl:02d}_{xtl:02d}_{xs}_{ys}'
                    # if buffer file is a valid sub-tile within the DEM
                    # if file doesn't exist: all fill value with all mask
                    if bkey in tile_attrs.keys():
                        bsub = tile_attrs[bkey]["tile"]
                        btar = f'{tile_attrs[bkey]["name"]}.tar.gz'
                        buffer_file = elevation_directory.joinpath(bkey, btar)
                        if not buffer_file.exists():
                            all_tiles.append(bsub)

    # sort and condense list
    print(','.join(sorted(set(all_tiles))))

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Determines which digital elevation model tiles
            to read for a given ICESat-2 ATL06 file
            """
    )
    # command line parameters
    parser.add_argument('file',
        type=pathlib.Path, nargs='+',
        help='ICESat-2 ATL06 file to run')
    # working data directory for location of DEM files
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    # Digital elevation model (REMA, ArcticDEM, GIMP) to run
    # set the DEM model to run for a given granule (else set automatically)
    parser.add_argument('--model','-m',
        metavar='DEM', type=str, choices=('REMA', 'ArcticDEM', 'GIMP'),
        help='Digital Elevation Model to run')
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
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

    # run program with parameters for each file
    for FILE in args.file:
        check_DEM_ICESat2_ATL06(FILE,
            DIRECTORY=args.directory,
            DEM_MODEL=args.model)

# run main program
if __name__ == '__main__':
    main()
