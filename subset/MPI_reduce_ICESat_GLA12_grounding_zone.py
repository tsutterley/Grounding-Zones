#!/usr/bin/env python
u"""
MPI_reduce_ICESat_GLA12_grounding_zone.py
Written by Tyler Sutterley (05/2024)

Create masks for reducing ICESat/GLAS L2 GLA12 Antarctic and Greenland
    Ice Sheet elevation data to within a buffer region near the ice
    sheet grounding zone
Used to calculate a more definite grounding zone from the ICESat data

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
    -H X, --hemisphere X: Region of interest to run
    -B X, --buffer X: Distance in kilometers to buffer from grounding line
    -p X, --polygon X: Georeferenced file containing a set of polygons
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
    mpi4py: MPI for Python
        http://pythonhosted.org/mpi4py/
        http://mpi4py.readthedocs.org/en/stable/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
        http://docs.h5py.org/en/stable/mpi.html
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
    Written 05/2024
"""
from __future__ import print_function

import sys
import os
import re
import logging
import pathlib
import argparse
import numpy as np
import grounding_zones as gz

# attempt imports
fiona = gz.utilities.import_dependency('fiona')
h5py = gz.utilities.import_dependency('h5py')
MPI = gz.utilities.import_dependency('mpi4py.MPI')
pyproj = gz.utilities.import_dependency('pyproj')
pyTMD = gz.utilities.import_dependency('pyTMD')
geometry = gz.utilities.import_dependency('shapely.geometry')
timescale = gz.utilities.import_dependency('timescale')

# buffered shapefile
buffer_shapefile = {}
buffer_shapefile['N'] = 'grn_ice_sheet_buffer_{0:0.0f}km.shp'
buffer_shapefile['S'] = 'ant_ice_sheet_islands_v2_buffer_{0:0.0f}km.shp'
# description and reference for each grounded ice file
grounded_description = {}
grounded_description['N'] = 'Greenland Mapping Project (GIMP) Ice & Ocean Mask'
grounded_description['S'] = ('MEaSUREs Antarctic Boundaries for IPY 2007-2009 '
    'from Satellite_Radar, Version 2')
grounded_reference = {}
grounded_reference['N'] = 'https://doi.org/10.5194/tc-8-1509-2014'
grounded_reference['S'] = 'https://doi.org/10.5067/IKBWW4RYHF1Q'

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
        description="""Create masks for reducing ICESat/GLAS L2
            GLA12 Antarctic and Greenland Ice Sheet elevation data
            to within a buffer region near the ice sheet grounding zone
            """
    )
    # command line parameters
    parser.add_argument('file',
        type=pathlib.Path,
        help='ICESat GLA12 file to run')
    # working data directory for shapefiles
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
    # buffer in kilometers for extracting grounding zone
    parser.add_argument('--buffer','-B',
        type=float, default=20.0,
        help='Distance in kilometers to buffer grounding zone')
    # alternatively read a specific georeferenced file
    parser.add_argument('--polygon','-p',
        type=pathlib.Path, default=None,
        help='Georeferenced file containing a set of polygons')
    # verbosity settings
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output files')
    # return the parser
    return parser

# PURPOSE: load the polygon object for the buffered estimated grounding zone
def load_grounding_zone(base_dir, HEM, BUFFER, shapefile=None):
    # buffered shapefile for region
    if shapefile is None:
        shapefile = buffer_shapefile[HEM].format(BUFFER)
        input_shapefile = base_dir.joinpath(shapefile)
    else:
        input_shapefile = pathlib.Path(shapefile).expanduser().absolute()
    # read buffered shapefile
    logging.info(str(input_shapefile))
    shape = fiona.open(str(input_shapefile))
    # extract coordinate reference system
    if ('init' in shape.crs.keys()):
        epsg = pyproj.CRS(shape.crs['init']).to_epsg()
    else:
        epsg = pyproj.CRS(shape.crs).to_epsg()
    # create list of polygons
    polygons = []
    # extract the entities and assign by tile name
    for i,ent in enumerate(shape.values()):
        # list of coordinates
        poly_list = []
        # extract coordinates for entity
        for coords in ent['geometry']['coordinates']:
            # extract Polar-Stereographic coordinates for record
            x,y = np.transpose(coords)
            poly_list.append(list(zip(x,y)))
        # convert poly_list into Polygon object with holes
        poly_obj = geometry.Polygon(poly_list[0], holes=poly_list[1:])
        # Valid Polygon cannot have overlapping exterior or interior rings
        if (not poly_obj.is_valid):
            poly_obj = poly_obj.buffer(0)
        polygons.append(poly_obj)
    # create shapely multipolygon object
    mpoly_obj = geometry.MultiPolygon(polygons)
    # close the shapefile
    shape.close()
    # return the polygon object for the ice sheet
    return (mpoly_obj, epsg)

# PURPOSE: read ICESat ice sheet HDF5 elevation data (GLAH12)
# reduce data to within buffer of grounding zone
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
    info(comm.rank, comm.size)
    if (comm.rank == 0):
        logging.info(f'{str(args.file)} -->')
    # input granule basename
    GRANULE = args.file.name

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
    VAR = 'GROUNDING_ZONE_MASK'
    try:
        PRD,RL,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE = \
            rx.findall(GRANULE).pop()
    except (ValueError, IndexError):
        # output mask HDF5 file (generic)
        FILENAME = f'{args.file.stem}_{VAR}{args.file.suffix}'
    else:
        # output mask HDF5 file for NSIDC granules
        fargs = (PRD,RL,VAR,RGTP,ORB,INST,CYCL,TRK,SEG,GRAN,TYPE)
        file_format = 'GLAH{0}_{1}_{2}_{3}{4}{5}_{6}_{7}_{8}_{9}_{10}.h5'
        FILENAME = file_format.format(*fargs)

    # get output directory from input file
    if args.output_directory is None:
        args.output_directory = args.file.parent

    # check if data is an s3 presigned url
    if str(args.file).startswith('s3:'):
        client = gz.utilities.attempt_login('urs.earthdata.nasa.gov',
            authorization_header=True)
        session = gz.utilities.s3_filesystem()
        args.file = session.open(args.file, mode='rb')
    else:
        args.file = pathlib.Path(args.file).expanduser().absolute()

    # Open the HDF5 file for reading
    fileID = h5py.File(args.file, 'r', driver='mpio', comm=comm)

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

    # parameters for Topex/Poseidon and WGS84 ellipsoids
    topex = pyTMD.datum(ellipsoid='TOPEX', units='MKS')
    wgs84 = pyTMD.datum(ellipsoid='WGS84', units='MKS')
    # convert from Topex/Poseidon to WGS84 Ellipsoids
    lat_40HZ, elev_40HZ = pyTMD.spatial.convert_ellipsoid(lat_TPX, elev_TPX,
        topex.a_axis, topex.flat, wgs84.a_axis, wgs84.flat,
        eps=1e-12, itmax=10)

    # read data on rank 0
    if (comm.rank == 0):
        # read shapefile and create shapely multipolygon objects
        mpoly_obj, epsg = load_grounding_zone(args.directory,
            args.hemisphere, args.buffer, shapefile=args.polygon)
    else:
        # create empty object for list of shapely objects
        mpoly_obj = None
        epsg = None

    # Broadcast Shapely multipolygon objects and projection
    mpoly_obj = comm.bcast(mpoly_obj, root=0)
    epsg = comm.bcast(epsg, root=0)

    # pyproj transformer for converting lat/lon to polar stereographic
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(epsg)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # define indices to run for specific process
    ind = np.arange(comm.Get_rank(), n_40HZ, comm.Get_size(), dtype=int)

    # convert lat/lon to polar stereographic
    X,Y = transformer.transform(lon_40HZ[ind], lat_40HZ[ind])
    # convert reduced x and y to shapely multipoint object
    xy_point = geometry.MultiPoint(np.c_[X, Y])

    # create distributed intersection map for calculation
    distributed_map = np.zeros((n_40HZ), dtype=bool)
    # create empty intersection map array for receiving
    associated_map = np.zeros((n_40HZ), dtype=bool)
    # for each polygon
    for poly_obj in mpoly_obj.geoms:
        # finds if points are encapsulated (in grounding zone)
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
        recvbuf=[associated_map, MPI.BOOL], op=MPI.LOR)
    distributed_map = None
    # wait for all processes to finish calculation
    comm.Barrier()
    # validity check
    valid_check = np.any(associated_map)

    # copy variables for outputting to HDF5 file
    IS_gla12_mask = dict(Data_40HZ={})
    IS_gla12_mask_attrs = dict(Data_40HZ={})

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
        IS_gla12_mask_attrs[att] = fileID.attrs[att]
    # copy ICESat campaign name from ancillary data
    IS_gla12_mask_attrs['Campaign'] = fileID['ANCILLARY_DATA'].attrs['Campaign']

    # add attributes for input GLA12 file
    IS_gla12_mask_attrs['lineage'] = pathlib.Path(args.file).name
    # update geospatial ranges for ellipsoid
    IS_gla12_mask_attrs['geospatial_lat_min'] = np.min(lat_40HZ)
    IS_gla12_mask_attrs['geospatial_lat_max'] = np.max(lat_40HZ)
    IS_gla12_mask_attrs['geospatial_lon_min'] = np.min(lon_40HZ)
    IS_gla12_mask_attrs['geospatial_lon_max'] = np.max(lon_40HZ)
    IS_gla12_mask_attrs['geospatial_lat_units'] = "degrees_north"
    IS_gla12_mask_attrs['geospatial_lon_units'] = "degrees_east"
    IS_gla12_mask_attrs['geospatial_ellipsoid'] = "WGS84"

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_mask_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time, geolocation and geophysical groups
    for var in ['Time','Geolocation','Geophysical']:
        IS_gla12_mask['Data_40HZ'][var] = {}
        IS_gla12_mask_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_mask_attrs['Data_40HZ'][var][att_name] = att_val

    # copy 40Hz group attributes
    for att_name,att_val in fileID['Data_40HZ'].attrs.items():
        IS_gla12_mask_attrs['Data_40HZ'][att_name] = att_val
    # copy attributes for time and geolocation groups
    for var in ['Time','Geolocation']:
        IS_gla12_mask['Data_40HZ'][var] = {}
        IS_gla12_mask_attrs['Data_40HZ'][var] = {}
        for att_name,att_val in fileID['Data_40HZ'][var].attrs.items():
            IS_gla12_mask_attrs['Data_40HZ'][var][att_name] = att_val

    # J2000 time
    IS_gla12_mask['Data_40HZ']['DS_UTCTime_40'] = DS_UTCTime_40HZ
    IS_gla12_mask_attrs['Data_40HZ']['DS_UTCTime_40'] = {}
    for att_name,att_val in fileID['Data_40HZ']['DS_UTCTime_40'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_mask_attrs['Data_40HZ']['DS_UTCTime_40'][att_name] = att_val
    # record
    IS_gla12_mask['Data_40HZ']['Time']['i_rec_ndx'] = rec_ndx_40HZ
    IS_gla12_mask_attrs['Data_40HZ']['Time']['i_rec_ndx'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Time']['i_rec_ndx']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Time']['i_rec_ndx'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_mask_attrs['Data_40HZ']['Time']['i_rec_ndx'][att_name] = att_val
    # latitude
    IS_gla12_mask['Data_40HZ']['Geolocation']['d_lat'] = lat_40HZ
    IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lat'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lat']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lat'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lat'][att_name] = att_val
    # longitude
    IS_gla12_mask['Data_40HZ']['Geolocation']['d_lon'] = lon_40HZ
    IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lon'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lon']['coordinates'] = \
        "../DS_UTCTime_40"
    for att_name,att_val in fileID['Data_40HZ']['Geolocation']['d_lon'].attrs.items():
        if att_name not in ('DIMENSION_LIST','CLASS','NAME'):
            IS_gla12_mask_attrs['Data_40HZ']['Geolocation']['d_lon'][att_name] = att_val

    # subsetting variables
    IS_gla12_mask['Data_40HZ']['Subsetting'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['Description'] = \
        ("The subsetting group contains parameters used to reduce values "
        "to specific regions of interest.")

    # output mask
    IS_gla12_mask['Data_40HZ']['Subsetting']['d_ice_gz'] = associated_map
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_ice_gz'] = {}
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_ice_gz']['contentType'] = \
        "referenceInformation"
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_ice_gz']['long_name'] = \
        'Grounding Zone Mask'
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_ice_gz']['description'] = \
        f"Grounding zone mask buffered by {args.buffer:0.0f} km"
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_ice_gz']['reference'] = \
        grounded_reference[args.hemisphere]
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_ice_gz']['source'] = \
        grounded_description[args.hemisphere]
    IS_gla12_mask_attrs['Data_40HZ']['Subsetting']['d_ice_gz']['coordinates'] = \
        "../DS_UTCTime_40"
    # wait for all processes to finish calculation
    comm.Barrier()

    # parallel h5py I/O does not support compression filters at this time
    if (comm.rank == 0) and valid_check:
        # output HDF5 files with output masks
        output_file = args.output_directory.joinpath(FILENAME)
        # print file information
        logging.info(f'\t{str(output_file)}')
        # write to output HDF5 file
        HDF5_GLA12_mask_write(IS_gla12_mask, IS_gla12_mask_attrs,
            FILENAME=output_file, CLOBBER=True)
        # change the permissions mode
        output_file.chmod(mode=args.mode)
    # close the input file
    fileID.close()

# PURPOSE: outputting the mask values for ICESat data to HDF5
def HDF5_GLA12_mask_write(IS_gla12_mask, IS_gla12_attrs,
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
    h5 = dict(Data_40HZ={})

    # add HDF5 file attributes
    attrs = {a:v for a,v in IS_gla12_attrs.items() if not isinstance(v,dict)}
    for att_name,att_val in attrs.items():
       fileID.attrs[att_name] = att_val

    # add software information
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version

    # create Data_40HZ group
    fileID.create_group('Data_40HZ')
    # add HDF5 40HZ group attributes
    for att_name,att_val in IS_gla12_attrs['Data_40HZ'].items():
        if att_name not in ('DS_UTCTime_40',) and not isinstance(att_val,dict):
            fileID['Data_40HZ'].attrs[att_name] = att_val

    # add 40HZ time variable
    val = IS_gla12_mask['Data_40HZ']['DS_UTCTime_40']
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
    for group in ['Time','Geolocation','Subsetting']:
        # add group to dict
        h5['Data_40HZ'][group] = {}
        # create Data_40HZ group
        fileID.create_group(f'Data_40HZ/{group}')
        # add HDF5 group attributes
        for att_name,att_val in IS_gla12_attrs['Data_40HZ'][group].items():
            if not isinstance(att_val,dict):
                fileID['Data_40HZ'][group].attrs[att_name] = att_val
        # for each variable in the group
        for key,val in IS_gla12_mask['Data_40HZ'][group].items():
            attrs = IS_gla12_attrs['Data_40HZ'][group][key]
            # Defining the HDF5 dataset variables
            var = f'Data_40HZ/{group}/{key}'
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
