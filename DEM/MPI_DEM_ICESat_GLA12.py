#!/usr/bin/env python
u"""
MPI_DEM_ICESat_GLA12.py
Written by Tyler Sutterley (06/2024)
Determines which digital elevation model tiles to read for a given GLA12 file
Reads 3x3 array of tiles for points within bounding box of central mosaic tile
Interpolates digital elevation model to locations of ICESat/GLAS L2
    GLA12 Antarctic and Greenland Ice Sheet elevation data

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
    -m X, --model X: Digital elevation model (REMA, ArcticDEM, GIMP) to run
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

REQUIRES MPI PROGRAM
    MPI: standardized and portable message-passing system
        https://www.open-mpi.org/
        http://mpitutorial.com/

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    mpi4py: MPI for Python
        http://pythonhosted.org/mpi4py/
        http://mpi4py.readthedocs.org/en/stable/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org
        http://docs.h5py.org/en/stable/mpi.html
    fiona: Python wrapper for vector data access functions from the OGR library
        https://fiona.readthedocs.io/en/latest/manual.html
    gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
        https://pypi.python.org/pypi/GDAL/
    shapely: PostGIS-ish operations outside a database context for Python
        http://toblerity.org/shapely/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    pyTMD: Python-based tidal prediction software
        https://pypi.org/project/pyTMD/
        https://pytmd.readthedocs.io/en/latest/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

REFERENCES:
    https://www.pgc.umn.edu/guides/arcticdem/data-description/
    https://www.pgc.umn.edu/guides/rema/data-description/
    https://nsidc.org/data/nsidc-0645/versions/1

UPDATE HISTORY:
    Written 06/2024
"""
from __future__ import print_function

import sys
import os
import re
import logging
import pathlib
import tarfile
import argparse
import numpy as np
import scipy.interpolate
import grounding_zones as gz

# attempt imports
fiona = gz.utilities.import_dependency('fiona')
gdal = gz.utilities.import_dependency('osgeo.gdal')
h5py = gz.utilities.import_dependency('h5py')
MPI = gz.utilities.import_dependency('mpi4py.MPI')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')
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

# PURPOSE: keep track of MPI threads
def info(rank, size):
    logging.info(f'Rank {rank+1:d} of {size:d}')
    logging.info(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.info(f'parent process: {os.getppid():d}')
    logging.info(f'process id: {os.getpid():d}')

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Interpolate DEMs to ICESat/GLAS L2 GLA12 Antarctic
            and Greenland Ice Sheet elevation data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = gz.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('file',
        type=pathlib.Path,
        help='ICESat GLA12 file to run')
    # working data directory for location of DEM files
    parser.add_argument('--directory','-D',
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help='Working data directory')
    # directory with output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # Digital elevation model (REMA, ArcticDEM, GIMP) to run
    # set the DEM model to run for a given granule (else set automatically)
    parser.add_argument('--model','-m',
        metavar='DEM', type=str, choices=('REMA', 'ArcticDEM', 'GIMP'),
        help='Digital Elevation Model to run')
    # verbosity settings
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permissions mode of output files')
    # return the parser
    return parser

# PURPOSE: read zip file containing index shapefiles for finding DEM tiles
def read_DEM_index(index_file, DEM_MODEL):
    # read the compressed shapefile and extract entities
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

# PURPOSE: read DEM tile file from gzipped tar files
def read_DEM_file(elevation_file, nd_value):
    # open file with tarfile (read)
    tar = tarfile.open(name=elevation_file, mode='r:gz')
    # find dem geotiff file within tar file
    member, = [m for m in tar.getmembers() if re.search(r'dem\.tif',m.name)]
    # use GDAL virtual file systems to read dem
    mmap_name = f"/vsitar/{elevation_file}/{member.name}"
    ds = gdal.Open(mmap_name)
    # read data matrix
    im = ds.GetRasterBand(1).ReadAsArray()
    fill_value = ds.GetRasterBand(1).GetNoDataValue()
    fill_value = 0.0 if (fill_value is None) else fill_value
    # get dimensions
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    # create mask for finding invalid values
    mask = np.zeros((ysize,xsize),dtype=bool)
    indy,indx = np.nonzero((im == fill_value) | (~np.isfinite(im)) |
        (np.ceil(im) == np.ceil(fill_value)))
    mask[indy,indx] = True
    # verify that values are finite by replacing with nd_value
    im[indy,indx] = nd_value
    # get geotiff info
    info_geotiff = ds.GetGeoTransform()
    # calculate image extents
    xmin = info_geotiff[0]
    ymax = info_geotiff[3]
    xmax = xmin + (xsize-1)*info_geotiff[1]
    ymin = ymax + (ysize-1)*info_geotiff[5]
    # close files
    ds = None
    gdal.Unlink(mmap_name)
    tar.close()
    # create image x and y arrays
    xi = np.arange(xmin,xmax+info_geotiff[1],info_geotiff[1])
    yi = np.arange(ymax,ymin+info_geotiff[5],info_geotiff[5])
    # return values (flip y values to be monotonically increasing)
    return (im[::-1,:],mask[::-1,:],xi,yi[::-1])

# PURPOSE: read DEM tile file from gzipped tar files to buffer main tile
def read_DEM_buffer(elevation_file, xlimits, ylimits, nd_value):
    # open file with tarfile (read)
    tar = tarfile.open(name=elevation_file, mode='r:gz')
    # find dem geotiff file within tar file
    member, = [m for m in tar.getmembers() if re.search(r'dem\.tif',m.name)]
    # use GDAL virtual file systems to read dem
    mmap_name = f"/vsitar/{elevation_file}/{member.name}"
    ds = gdal.Open(mmap_name)
    # get geotiff info
    info_geotiff = ds.GetGeoTransform()
    # original image extents
    xmin = info_geotiff[0]
    ymax = info_geotiff[3]
    # reduce input image with GDAL
    # Specify offset and rows and columns to read
    xoffset = int((xlimits[0] - xmin)/info_geotiff[1])
    yoffset = int((ymax - ylimits[1])/np.abs(info_geotiff[5]))
    xcount = int((xlimits[1] - xlimits[0])/info_geotiff[1]) + 1
    ycount = int((ylimits[1] - ylimits[0])/np.abs(info_geotiff[5])) + 1
    # read data matrix
    im = ds.GetRasterBand(1).ReadAsArray(xoffset, yoffset, xcount, ycount)
    fill_value = ds.GetRasterBand(1).GetNoDataValue()
    fill_value = 0.0 if (fill_value is None) else fill_value
    # create mask for finding invalid values
    mask = np.zeros((ycount,xcount),dtype=bool)
    indy,indx = np.nonzero((im == fill_value) | (~np.isfinite(im)) |
        (np.ceil(im) == np.ceil(fill_value)))
    mask[indy,indx] = True
    # verify that values are finite by replacing with nd_value
    im[indy,indx] = nd_value
    # reduced x and y limits of image
    xmin_reduced = xmin + xoffset*info_geotiff[1]
    xmax_reduced = xmin + xoffset*info_geotiff[1] + (xcount-1)*info_geotiff[1]
    ymax_reduced = ymax + yoffset*info_geotiff[5]
    ymin_reduced = ymax + yoffset*info_geotiff[5] + (ycount-1)*info_geotiff[5]
    # close files
    ds = None
    gdal.Unlink(mmap_name)
    tar.close()
    # create image x and y arrays
    xi = np.arange(xmin_reduced,xmax_reduced+info_geotiff[1],info_geotiff[1])
    yi = np.arange(ymax_reduced,ymin_reduced+info_geotiff[5],info_geotiff[5])
    # return values (flip y values to be monotonically increasing)
    return (im[::-1,:],mask[::-1,:],xi,yi[::-1])

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12)
# interpolate DEM data to x and y coordinates
def main():
    # start MPI communicator
    comm = MPI.COMM_WORLD

    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # output module information for process
    info(comm.rank,comm.size)
    # input granule basename
    GRANULE = args.file.name
    if (comm.rank == 0):
        logging.info(f'{str(args.file)} -->')

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
    try:
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = rx.findall(GRANULE).pop()
    except:
        # output DEM HDF5 file (generic)
        FILENAME = f'{args.file.stem}_{args.model}_{args.file.suffix}'
    else:
        # output DEM HDF5 file for NSIDC granules
        args = (PRD,RL,args.model,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_{2}_{3}{4}{5}_{6}_{7}_{8}_{9}_{10}.h5'
        FILENAME = file_format.format(*args)
    # get output directory from input file
    if args.output_directory is None:
        args.output_directory = args.file.parent
    # full path to output file
    OUTPUT_FILE = args.output_directory.joinpath(FILENAME)

    # check if data is an s3 presigned url
    if str(args.file).startswith('s3:'):
        client = gz.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = gz.utilities.s3_filesystem()
        INPUT_FILE = session.open(args.file, mode='rb')
    else:
        INPUT_FILE = pathlib.Path(args.file).expanduser().absolute()

    # read data from input file
    # read GLAH12 HDF5 file
    fileID = h5py.File(INPUT_FILE, 'r', driver='mpio', comm=comm)

    # get variables and attributes
    n_40HZ, = fileID['Data_40HZ']['Time']['i_rec_ndx'].shape
    rec_ndx_40HZ = fileID['Data_40HZ']['Time']['i_rec_ndx'][:].copy()
    # seconds since 2000-01-01 12:00:00 UTC (J2000)
    DS_UTCTime_40HZ = fileID['Data_40HZ']['DS_UTCTime_40'][:].copy()
    # Latitude (degrees North)
    lat_TPX = fileID['Data_40HZ']['Geolocation']['d_lat'][:].copy()
    # Longitude (degrees East)
    lon_40HZ = fileID['Data_40HZ']['Geolocation']['d_lon'][:].copy()
    # Elevation (height above TOPEX/Poseidon ellipsoid in meters)
    elev_TPX = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'][:].copy()
    fv = fileID['Data_40HZ']['Elevation_Surfaces']['d_elev'].attrs['_FillValue']

    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = pyTMD.datum(ellipsoid='TOPEX', units='MKS')
    wgs84 = pyTMD.datum(ellipsoid='WGS84', units='MKS')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ,elev_40HZ = pyTMD.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        topex.a_axis, topex.flat, wgs84.a_axis, wgs84.flat,
        eps=1e-12, itmax=10)
    
    # regular expression pattern for extracting parameters from ArcticDEM name
    rx1 = re.compile(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+m)_(.*?)$', re.VERBOSE)
    # full path to DEM directory
    args.directory = pathlib.Path(args.directory).expanduser().absolute()
    elevation_directory = args.directory.joinpath(*elevation_dir[args.model])
    # zip file containing index shapefiles for finding DEM tiles
    index_file = elevation_directory.joinpath(elevation_tile_index[args.model])

    # read data on rank 0
    if (comm.rank == 0):
        # read index file for determining which tiles to read
        tile_dict,tile_attrs,tile_epsg = read_DEM_index(index_file, args.model)
    else:
        # create empty object for list of shapely objects
        tile_dict = None
        tile_attrs = None
        tile_epsg = None

    # Broadcast Shapely polygon objects
    tile_dict = comm.bcast(tile_dict, root=0)
    tile_attrs = comm.bcast(tile_attrs, root=0)
    tile_epsg = comm.bcast(tile_epsg, root=0)
    valid_tiles = False

    # define indices to run for specific process
    ind = np.arange(comm.Get_rank(), n_40HZ, comm.Get_size(), dtype=int)

    # pyproj transformer for converting from latitude/longitude
    # into DEM tile coordinates
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(tile_epsg)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # output interpolated digital elevation model
    distributed_dem = np.ma.zeros((n_40HZ),fill_value=fv,dtype=np.float32)
    distributed_dem.mask = np.ones((n_40HZ),dtype=bool)
    dem_h = np.ma.zeros((n_40HZ),fill_value=fv,dtype=np.float32)
    dem_h.mask = np.ones((n_40HZ),dtype=bool)
    # convert projection from latitude/longitude to tile EPSG
    X,Y = transformer.transform(lon_40HZ, lat_40HZ)

    # convert reduced x and y to shapely multipoint object
    xy_point = shapely.geometry.MultiPoint(np.c_[X[ind], Y[ind]])

    # create complete masks for each DEM tile
    associated_map = {}
    for key,poly_obj in tile_dict.items():
        # create empty intersection map array for distributing
        distributed_map = np.zeros((n_40HZ),dtype=int)
        # create empty intersection map array for receiving
        associated_map[key] = np.zeros((n_40HZ),dtype=int)
        # finds if points are encapsulated (within tile)
        int_test = poly_obj.intersects(xy_point)
        if int_test:
            # extract intersected points
            int_map = list(map(poly_obj.intersects, xy_point.geoms))
            int_indices, = np.nonzero(int_map)
            # set distributed_map indices to True for intersected points
            distributed_map[ind[int_indices]] = True
        # communicate output MPI matrices between ranks
        # operation is a logical "or" across the elements.
        comm.Allreduce(sendbuf=[distributed_map, MPI.BOOL], \
            recvbuf=[associated_map[key], MPI.BOOL], op=MPI.LOR)
        distributed_map = None
    # wait for all processes to finish calculation
    comm.Barrier()
    # find valid tiles and free up memory from invalid tiles
    valid_tiles = [k for k,v in associated_map.items() if v.any()]
    invalid_tiles = sorted(set(associated_map.keys()) - set(valid_tiles))
    for key in invalid_tiles:
        associated_map[key] = None

    # copy variables for outputting to HDF5 file
    IS_gla12_dem = dict(Data_40HZ={})
    IS_gla12_fill = dict(Data_40HZ={})
    IS_gla12_dem_attrs = dict(Data_40HZ={})

    # copy global file attributes
    global_attribute_list = ['featureType','title','comment','summary','license',
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
        IS_gla12_dem_attrs[att] = fileID.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_dem_attrs['Campaign'] = fileID['ANCILLARY_DATA'].attrs['Campaign']

    # add attributes for input GLA12 file
    IS_gla12_dem_attrs['lineage'] = pathlib.Path(args.file).name
    # update geospatial ranges for ellipsoid
    IS_gla12_dem_attrs['geospatial_lat_min'] = np.min(lat_40HZ)
    IS_gla12_dem_attrs['geospatial_lat_max'] = np.max(lat_40HZ)
    IS_gla12_dem_attrs['geospatial_lon_min'] = np.min(lon_40HZ)
    IS_gla12_dem_attrs['geospatial_lon_max'] = np.max(lon_40HZ)
    IS_gla12_dem_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_dem_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_dem_attrs['geospatial_ellipsoid'] = "WGS84"

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_dem_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time, geolocation and geophysical groups
    for var in ['Time','Geolocation','Geophysical']:
        IS_gla12_dem['Data_40HZ'][var] = {}
        IS_gla12_fill['Data_40HZ'][var] = {}
        IS_gla12_dem_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_dem_attrs['Data_40HZ'][var][att_name] = att_val
    # subsetting variables
    IS_gla12_dem['Data_40HZ']['Subsetting'] = {}
    IS_gla12_fill['Data_40HZ']['Subsetting'] = {}
    IS_gla12_dem_attrs['Data_40HZ']['Subsetting'] = {}
    IS_gla12_dem_attrs['Data_40HZ']['Subsetting']['Description']= \
        ("The subsetting group contains parameters used to reduce values "
        "to specific regions of interest.")
    
    # J2000 time
    IS_gla12_dem['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_fill['Data_40HZ']['DS_UTCTime_40'] = None
    IS_gla12_dem_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in fileID['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_dem_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    # record
    IS_gla12_dem['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_fill['Data_40HZ']['Time']['i_rec_ndx'] = None
    IS_gla12_dem_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_dem_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    # latitude
    IS_gla12_dem['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lat'] = None
    IS_gla12_dem_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_dem_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    # longitude
    IS_gla12_dem['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_fill['Data_40HZ']['Geolocation']['d_lon'] = None
    IS_gla12_dem_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_dem_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    # for each valid tile
    for key in valid_tiles:
        # output mask to HDF5
        sub = tile_attrs[key]['tile']
        IS_gla12_dem['Data_40HZ']['Subsetting'][key] = associated_map[key]
        IS_gla12_fill['Data_40HZ']['Subsetting'][key] = None
        IS_gla12_dem_attrs['Data_40HZ']['Subsetting'][key] = {}
        IS_gla12_dem_attrs['Data_40HZ']['Subsetting'][key]['long_name'] = \
            f'{key} Mask'
        IS_gla12_dem_attrs['Data_40HZ']['Subsetting'][key]['description'] = \
            f'Name of DEM tile {sub} encapsulating the land ice segments.'
        IS_gla12_dem_attrs['Data_40HZ']['Subsetting'][key]['source'] = args.model
        IS_gla12_dem_attrs['Data_40HZ']['Subsetting'][key]['coordinates'] = \
            "../DS_UTCTime_40"
        # add DEM attributes
        if args.model in ('REMA','ArcticDEM'):
            IS_gla12_dem_attrs['Data_40HZ']['Subsetting'][key]['meanresz'] = \
                tile_attrs[key]['meanresz']
            IS_gla12_dem_attrs['Data_40HZ']['Subsetting'][key]['num_gcps'] = \
                tile_attrs[key]['num_gcps']

    # read and interpolate DEM to coordinates in parallel
    for t in range(comm.Get_rank(), len(valid_tiles), comm.Get_size()):
        key = valid_tiles[t]
        sub = tile_attrs[key]['tile']
        name = tile_attrs[key]['name']
        # read central DEM file (geotiff within gzipped tar file)
        tar = f'{name}.tar.gz'
        elevation_file = elevation_directory.joinpath(sub,tar)
        DEM,MASK,xi,yi = read_DEM_file(elevation_file, fv)
        # buffer DEM using values from adjacent tiles
        # use 400m (10 geosegs and divisible by ArcticDEM and REMA pixels)
        # use 1500m for GIMP
        bf = 1500 if (args.model == 'GIMP') else 400
        ny,nx = np.shape(DEM)
        dx = np.abs(xi[1]-xi[0]).astype('i')
        dy = np.abs(yi[1]-yi[0]).astype('i')
        # new buffered DEM and mask
        d = np.full((ny+2*bf//dy,nx+2*bf//dx),fv,dtype=np.float32)
        m = np.ones((ny+2*bf//dy,nx+2*bf//dx),dtype=bool)
        d[bf//dy:-bf//dy,bf//dx:-bf//dx] = DEM.copy()
        m[bf//dy:-bf//dy,bf//dx:-bf//dx] = MASK.copy()
        DEM,MASK = (None,None)
        # new buffered image x and y coordinates
        x = (xi[0] - bf) + np.arange((nx+2*bf//dx))*dx
        y = (yi[0] - bf) + np.arange((ny+2*bf//dy))*dy
        # min and max of left column, center column, right column
        XL,XC,XR = [[xi[0]-bf,xi[0]-dx],[xi[0],xi[-1]],[xi[-1]+dx,xi[-1]+bf]]
        xlimits = [XL,XL,XL,XC,XC,XR,XR,XR] # LLLCCRRR
        # min and max of bottom row, middle row, top row
        YB,YM,YT = [[yi[0]-bf,yi[0]-dy],[yi[0],yi[-1]],[yi[-1]+dy,yi[-1]+bf]]
        ylimits = [YB,YM,YT,YB,YT,YB,YM,YT] # BMTBTBMT

        # buffer using neighbor tiles (REMA/GIMP) or sub-tiles (ArcticDEM)
        if (args.model == 'REMA'):
            # REMA tiles to read to buffer the image
            IMy,IMx=np.array(re.findall(r'(\d+)_(\d+)',sub).pop(),dtype='i')
            # neighboring tiles for buffering DEM (LB,LM,LT,CB,CT,RB,RM,RT)
            xtiles=[IMx-1,IMx-1,IMx-1,IMx,IMx,IMx+1,IMx+1,IMx+1] # LLLCCRRR
            ytiles=[IMy-1,IMy,IMy+1,IMy-1,IMy+1,IMy-1,IMy,IMy+1] # BMTBTBMT
            for xtl,ytl,xlim,ylim in zip(xtiles,ytiles,xlimits,ylimits):
                # read DEM file (geotiff within gzipped tar file)
                bkey = f'{ytl:02d}_{xtl:02d}'
                # if buffer file is a valid tile within the DEM
                # if file doesn't exist: will be all fill value with all mask
                if bkey in tile_attrs.keys():
                    bsub = tile_attrs[bkey]['tile']
                    bname = tile_attrs[bkey]['name']
                    btar = f'{bname}.tar.gz'
                    buffer_file = elevation_directory.joinpath(bkey,btar)
                    if not buffer_file.exists():
                        raise FileNotFoundError(f'{buffer_file} not found')
                    DEM,MASK,x1,y1=read_DEM_buffer(buffer_file,xlim,ylim,fv)
                    xmin = int((x1[0] - x[0])//dx)
                    xmax = int((x1[-1] - x[0])//dx) + 1
                    ymin = int((y1[0] - y[0])//dy)
                    ymax = int((y1[-1] - y[0])//dy) + 1
                    # add to buffered DEM and mask
                    d[ymin:ymax,xmin:xmax] = DEM.copy()
                    m[ymin:ymax,xmin:xmax] = MASK.copy()
                    DEM,MASK = (None,None)
        elif (args.model == 'GIMP'):
            # GIMP tiles to read to buffer the image
            IMx,IMy=np.array(re.findall(r'(\d+)_(\d+)',sub).pop(),dtype='i')
            # neighboring tiles for buffering DEM (LB,LM,LT,CB,CT,RB,RM,RT)
            xtiles=[IMx-1,IMx-1,IMx-1,IMx,IMx,IMx+1,IMx+1,IMx+1] # LLLCCRRR
            ytiles=[IMy-1,IMy,IMy+1,IMy-1,IMy+1,IMy-1,IMy,IMy+1] # BMTBTBMT
            for xtl,ytl,xlim,ylim in zip(xtiles,ytiles,xlimits,ylimits):
                # read DEM file (geotiff within gzipped tar file)
                bkey = f'{xtl:d}_{ytl:d}'
                # if buffer file is a valid tile within the DEM
                # if file doesn't exist: will be all fill value with all mask
                if bkey in tile_attrs.keys():
                    bsub = tile_attrs[bkey]['tile']
                    bname = tile_attrs[bkey]['name']
                    btar = f'{bname}.tar.gz'
                    buffer_file = elevation_directory.joinpath(bkey,btar)
                    if not buffer_file.exists():
                        raise FileNotFoundError(f'{buffer_file} not found')
                    DEM,MASK,x1,y1=read_DEM_buffer(buffer_file,xlim,ylim,fv)
                    xmin = int((x1[0] - x[0])//dx)
                    xmax = int((x1[-1] - x[0])//dx) + 1
                    ymin = int((y1[0] - y[0])//dy)
                    ymax = int((y1[-1] - y[0])//dy) + 1
                    # add to buffered DEM and mask
                    d[ymin:ymax,xmin:xmax] = DEM.copy()
                    m[ymin:ymax,xmin:xmax] = MASK.copy()
                    DEM,MASK = (None,None)
        elif (args.model == 'ArcticDEM'):
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
            kwargs = (xtiles,ytiles,xsubtiles,ysubtiles,xlimits,ylimits)
            for xtl,ytl,xs,ys,xlim,ylim in zip(*kwargs):
                # read DEM file (geotiff within gzipped tar file)
                bkey = f'{ytl:02d}_{xtl:02d}_{xs}_{ys}'
                # if buffer file is a valid sub-tile within the DEM
                # if file doesn't exist: all fill value with all mask
                if bkey in tile_attrs.keys():
                    bsub = tile_attrs[bkey]['tile']
                    bname = tile_attrs[bkey]['name']
                    btar = f'{bname}.tar.gz'
                    buffer_file = elevation_directory.joinpath(bsub,btar)
                    if not buffer_file.exists():
                        raise FileNotFoundError(f'{buffer_file} not found')
                    DEM,MASK,x1,y1=read_DEM_buffer(buffer_file,xlim,ylim,fv)
                    xmin = int((x1[0] - x[0])//dx)
                    xmax = int((x1[-1] - x[0])//dx) + 1
                    ymin = int((y1[0] - y[0])//dy)
                    ymax = int((y1[-1] - y[0])//dy) + 1
                    # add to buffered DEM and mask
                    d[ymin:ymax,xmin:xmax] = DEM.copy()
                    m[ymin:ymax,xmin:xmax] = MASK.copy()
                    DEM,MASK = (None,None)

        # indices of x and y coordinates within tile
        tile_indices, = np.nonzero(associated_map[key])
        # use spline interpolation to calculate DEM values at coordinates
        f1 = scipy.interpolate.RectBivariateSpline(x,y,d.T,kx=1,ky=1)
        f2 = scipy.interpolate.RectBivariateSpline(x,y,m.T,kx=1,ky=1)
        dataout = f1.ev(X[tile_indices],Y[tile_indices])
        maskout = f2.ev(X[tile_indices],Y[tile_indices])
        # save DEM to output variables
        distributed_dem.data[tile_indices] = dataout
        distributed_dem.mask[tile_indices] = maskout.astype(bool)
        # clear DEM and mask variables
        f1,f2,dataout,maskout,d,m = (None,None,None,None,None,None)

    # communicate output MPI matrices between ranks
    # operations are element summations and logical "and" across elements
    comm.Allreduce(sendbuf=[distributed_dem.data, MPI.FLOAT], \
        recvbuf=[dem_h.data, MPI.FLOAT], op=MPI.SUM)
    comm.Allreduce(sendbuf=[distributed_dem.mask, MPI.BOOL], \
        recvbuf=[dem_h.mask, MPI.BOOL], op=MPI.LAND)
    distributed_dem = None
    # wait for all distributed processes to finish for beam
    comm.Barrier()

    # output interpolated DEM to HDF5
    dem_h.mask[np.abs(dem_h.data) >= 1e4] = True
    dem_h.data[dem_h.mask] = dem_h.fill_value
    IS_gla12_dem['Data_40HZ']['Geophysical']['d_DEM_elv'] = dem_h
    IS_gla12_fill['Data_40HZ']['Geophysical']['d_DEM_elv'] = dem_h.fill_value
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv'] = {}
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['units'] = "meters"
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['long_name'] = \
        "DEM Height"
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['description'] = \
        ("Height of the DEM, interpolated by bivariate-spline interpolation in the "
        "DEM coordinate system to the segment location.")
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['source'] = args.model
    IS_gla12_dem_attrs['Data_40HZ']['Geophysical']['d_DEM_elv']['coordinates'] = \
        "../DS_UTCTime_40"

    # parallel h5py I/O does not support compression filters at this time
    if (comm.rank == 0) and bool(valid_tiles):
        # print file information
        logging.info(f'\t{OUTPUT_FILE}')
        HDF5_GLA12_dem_write(IS_gla12_dem, IS_gla12_dem_attrs,
            FILENAME=OUTPUT_FILE,
            FILL_VALUE=IS_gla12_fill,
            INPUT=GRANULE,
            CLOBBER=True)
        # change the permissions mode
        OUTPUT_FILE.chmod(mode=args.mode)
    # close the input file
    fileID.close()

# PURPOSE: outputting the DEM values for ICESat data to HDF5
def HDF5_GLA12_dem_write(IS_gla12_tide, IS_gla12_attrs,
    FILENAME='', INPUT=[], FILL_VALUE=None, CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    FILENAME = pathlib.Path(FILENAME).expanduser().absolute()
    fileID = h5py.File(FILENAME, clobber)
    # create 40HZ HDF5 records
    h5 = dict(Data_40HZ={})

    # add HDF5 file attributes
    attrs = {a:v for a,v in IS_gla12_attrs.items() if not isinstance(v,dict)}
    for att_name,att_val in attrs.items():
       fileID.attrs[att_name] = att_val

    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version
    # add attributes for input files
    fileID.attrs['lineage'] = pathlib.Path(INPUT).name

    # create Data_40HZ group
    fileID.create_group('Data_40HZ')
    # add HDF5 40HZ group attributes
    for att_name,att_val in IS_gla12_attrs['Data_40HZ'].items():
        if att_name not in ('DS_UTCTime_40',) and not isinstance(att_val,dict):
            fileID['Data_40HZ'].attrs[att_name] = att_val

    # add 40HZ time variable
    val = IS_gla12_tide['Data_40HZ']['DS_UTCTime_40']
    attrs = IS_gla12_attrs['Data_40HZ']['DS_UTCTime_40']
    # Defining the HDF5 dataset variables
    h5['Data_40HZ']['DS_UTCTime_40'] = fileID.create_dataset(
        'Data_40HZ/DS_UTCTime_40', np.shape(val),
        data=val, dtype=val.dtype, compression='gzip')
    # make dimension
    h5['Data_40HZ']['DS_UTCTime_40'].make_scale('DS_UTCTime_40')
    # add HDF5 variable attributes
    for att_name,att_val in attrs.items():
        h5['Data_40HZ']['DS_UTCTime_40'].attrs[att_name] = att_val

    # for each variable group
    for group in ['Time','Geolocation','Geophysical','Subsetting']:
        # add group to dict
        h5['Data_40HZ'][group] = {}
        # create Data_40HZ group
        fileID.create_group(f'Data_40HZ/{group}')
        # add HDF5 group attributes
        for att_name,att_val in IS_gla12_attrs['Data_40HZ'][group].items():
            if not isinstance(att_val,dict):
                fileID['Data_40HZ'][group].attrs[att_name] = att_val
        # for each variable in the group
        for key,val in IS_gla12_tide['Data_40HZ'][group].items():
            fillvalue = FILL_VALUE['Data_40HZ'][group][key]
            attrs = IS_gla12_attrs['Data_40HZ'][group][key]
            # Defining the HDF5 dataset variables
            var = f'Data_40HZ/{group}/{key}'
            # use variable compression if containing fill values
            if fillvalue:
                h5['Data_40HZ'][group][key] = fileID.create_dataset(var,
                    np.shape(val), data=val, dtype=val.dtype,
                    fillvalue=fillvalue, compression='gzip')
            else:
                h5['Data_40HZ'][group][key] = fileID.create_dataset(var,
                    np.shape(val), data=val, dtype=val.dtype,
                    compression='gzip')
            # attach dimensions
            for i,dim in enumerate(['DS_UTCTime_40']):
                h5['Data_40HZ'][group][key].dims[i].attach_scale(
                    h5['Data_40HZ'][dim])
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5['Data_40HZ'][group][key].attrs[att_name] = att_val

    # Closing the HDF5 file
    fileID.close()

# run main program
if __name__ == '__main__':
    main()
