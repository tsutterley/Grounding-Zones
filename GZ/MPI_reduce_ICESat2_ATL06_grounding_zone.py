#!/usr/bin/env python
u"""
MPI_reduce_ICESat2_ATL06_grounding_zone.py
Written by Tyler Sutterley (08/2023)

Create masks for reducing ICESat-2 land ice height data to within
    a buffer region near the ice sheet grounding zone
Used to calculate a more definite grounding zone from the ICESat-2 data

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O X, --output-directory X: input/output data directory
    -B X, --buffer X: Distance in kilometers to buffer from grounding line
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

PROGRAM DEPENDENCIES:
    convert_delta_time.py: converts from delta time into Julian and year-decimal
    time.py: Utilities for calculating time operations
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 08/2023: create s3 filesystem when using s3 urls as input
        use time functions from timescale.time
    Updated 07/2023: using pathlib to define and operate on paths
        use geoms attribute for shapely 2.0 compliance
    Updated 12/2022: single implicit import of grounding zone tools
    Updated 11/2022: verify coordinate reference system of shapefile
    Updated 10/2022: simplied HDF5 file output to match other reduction programs
    Updated 08/2022: use logging for verbose output of processing run
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: use argparse descriptions within documentation
    Updated 02/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 01/2021: time utilities for converting times from JD and to decimal
    Updated 12/2020: H5py deprecation warning change to use make_scale
    Updated 10/2020: using argparse to set parameters.  update pyproj transforms
    Updated 08/2020: using convert delta time function to convert to Julian days
    Updated 09/2019: using fiona for shapefile read and pyproj for coordinates
    Updated 05/2019: check if beam exists in a try except else clause
    Updated 04/2019: check if subsetted beam contains land ice data
    Forked 04/2019 from MPI_reduce_triangulated_grounding_zone.py
    Updated 02/2019: shapely updates for python3 compatibility
    Updated 07/2017: using parts from shapefile
    Written 06/2017
"""
from __future__ import print_function

import sys
import os
import re
import logging
import pathlib
import argparse
import datetime
import warnings
import numpy as np
import grounding_zones as gz

# attempt imports
try:
    import fiona
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("fiona not available", ImportWarning)
try:
    import h5py
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("h5py not available", ImportWarning)
try:
    from mpi4py import MPI
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("mpi4py not available", ImportWarning)
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyproj not available", ImportWarning)
try:
    import shapely.geometry
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("shapely not available", ImportWarning)
try:
    import timescale
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("timescale not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

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
        description="""Create masks for reducing ICESat-2 land ice height
            data to within a buffer region near the ice sheet grounding zone
            """
    )
    # command line parameters
    parser.add_argument('file',
        type=pathlib.Path,
        help='ICESat-2 ATL06 file to run')
    # working data directory for shapefiles
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=gz.utilities.get_data_path('data'),
        help='Working data directory')
    # directory with input/output data
    parser.add_argument('--output-directory','-O',
        type=pathlib.Path,
        help='Output data directory')
    # buffer in kilometers for extracting grounding zone
    parser.add_argument('--buffer','-B',
        type=float, default=20.0,
        help='Distance in kilometers to buffer grounding zone')
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

# PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        projection_flag = 'S'
    elif GRANULE in ('03','04','05'):
        projection_flag = 'N'
    return projection_flag

# PURPOSE: load the polygon object for the buffered estimated grounding zone
def load_grounding_zone(base_dir, HEM, BUFFER):
    # buffered shapefile for region
    input_shapefile = base_dir.joinpath(buffer_shapefile[HEM].format(BUFFER))
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
        poly_obj = shapely.geometry.Polygon(poly_list[0],holes=poly_list[1:])
        # Valid Polygon cannot have overlapping exterior or interior rings
        if (not poly_obj.is_valid):
            poly_obj = poly_obj.buffer(0)
        polygons.append(poly_obj)
    # create shapely multipolygon object
    mpoly_obj = shapely.geometry.MultiPolygon(polygons)
    # close the shapefile
    shape.close()
    # return the polygon object for the ice sheet
    return (mpoly_obj, epsg)

# PURPOSE: read ICESat-2 land ice height data (ATL06)
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

    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS,AUX = rx.findall(GRANULE).pop()
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

    # set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRN)
    # read each input beam within the file
    IS2_atl06_beams = []
    for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]',k))]:
        # check if subsetted beam contains land ice data
        try:
            fileID[gtx]['land_ice_segments']['segment_id']
        except KeyError:
            pass
        else:
            IS2_atl06_beams.append(gtx)

    # read data on rank 0
    if (comm.rank == 0):
        # read shapefile and create shapely multipolygon objects
        mpoly_obj, epsg = load_grounding_zone(args.directory, HEM, args.buffer)
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

    # copy variables for outputting to HDF5 file
    IS2_atl06_mask = {}
    IS2_atl06_fill = {}
    IS2_atl06_dims = {}
    IS2_atl06_mask_attrs = {}
    # combined validity check for all beams
    valid_check = False
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl06_mask['ancillary_data'] = {}
    IS2_atl06_mask_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl06_mask['ancillary_data'][key] = fileID['ancillary_data'][key][:]
        # Getting attributes of group and included variables
        IS2_atl06_mask_attrs['ancillary_data'][key] = {}
        for att_name,att_val in fileID['ancillary_data'][key].attrs.items():
            IS2_atl06_mask_attrs['ancillary_data'][key][att_name] = att_val

    # for each input beam within the file
    for gtx in sorted(IS2_atl06_beams):
        # output data dictionaries for beam
        IS2_atl06_mask[gtx] = dict(land_ice_segments={})
        IS2_atl06_fill[gtx] = dict(land_ice_segments={})
        IS2_atl06_dims[gtx] = dict(land_ice_segments={})
        IS2_atl06_mask_attrs[gtx] = dict(land_ice_segments={})

        # number of segments
        segment_id = fileID[gtx]['land_ice_segments']['segment_id'][:]
        n_seg, = fileID[gtx]['land_ice_segments']['segment_id'].shape
        # invalid value for beam
        fv = fileID[gtx]['land_ice_segments']['h_li'].fillvalue

        # define indices to run for specific process
        ind = np.arange(comm.Get_rank(), n_seg, comm.Get_size(), dtype=int)

        # extract delta time
        delta_time = fileID[gtx]['land_ice_segments']['delta_time'][:].copy()
        # extract lat/lon
        longitude = fileID[gtx]['land_ice_segments']['longitude'][:].copy()
        latitude = fileID[gtx]['land_ice_segments']['latitude'][:].copy()
        # convert lat/lon to polar stereographic
        X,Y = transformer.transform(longitude[ind], latitude[ind])
        # convert reduced x and y to shapely multipoint object
        xy_point = shapely.geometry.MultiPoint(np.c_[X, Y])

        # create distributed intersection map for calculation
        distributed_map = np.zeros((n_seg),dtype=bool)
        # create empty intersection map array for receiving
        associated_map = np.zeros((n_seg),dtype=bool)
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
        # add to validity check
        valid_check |= np.any(associated_map)

        # group attributes for beam
        IS2_atl06_mask_attrs[gtx]['Description'] = fileID[gtx].attrs['Description']
        IS2_atl06_mask_attrs[gtx]['atlas_pce'] = fileID[gtx].attrs['atlas_pce']
        IS2_atl06_mask_attrs[gtx]['atlas_beam_type'] = fileID[gtx].attrs['atlas_beam_type']
        IS2_atl06_mask_attrs[gtx]['groundtrack_id'] = fileID[gtx].attrs['groundtrack_id']
        IS2_atl06_mask_attrs[gtx]['atmosphere_profile'] = fileID[gtx].attrs['atmosphere_profile']
        IS2_atl06_mask_attrs[gtx]['atlas_spot_number'] = fileID[gtx].attrs['atlas_spot_number']
        IS2_atl06_mask_attrs[gtx]['sc_orientation'] = fileID[gtx].attrs['sc_orientation']
        # group attributes for land_ice_segments
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['Description'] = ("The land_ice_segments group "
            "contains the primary set of derived products. This includes geolocation, height, and "
            "standard error and quality measures for each segment. This group is sparse, meaning "
            "that parameters are provided only for pairs of segments for which at least one beam "
            "has a valid surface-height measurement.")
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['data_rate'] = ("Data within this group are "
            "sparse.  Data values are provided only for those ICESat-2 20m segments where at "
            "least one beam has a valid land ice height measurement.")

        # geolocation, time and segment ID
        # delta time
        IS2_atl06_mask[gtx]['land_ice_segments']['delta_time'] = delta_time
        IS2_atl06_fill[gtx]['land_ice_segments']['delta_time'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['delta_time'] = None
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['delta_time'] = {}
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['delta_time']['standard_name'] = "time"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['delta_time']['calendar'] = "standard"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['delta_time']['coordinates'] = \
            "segment_id latitude longitude"
        # latitude
        IS2_atl06_mask[gtx]['land_ice_segments']['latitude'] = latitude
        IS2_atl06_fill[gtx]['land_ice_segments']['latitude'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['latitude'] = ['delta_time']
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude'] = {}
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude']['units'] = "degrees_north"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude']['valid_min'] = -90.0
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude']['valid_max'] = 90.0
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['latitude']['coordinates'] = \
            "segment_id delta_time longitude"
        # longitude
        IS2_atl06_mask[gtx]['land_ice_segments']['longitude'] = longitude
        IS2_atl06_fill[gtx]['land_ice_segments']['longitude'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['longitude'] = ['delta_time']
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude'] = {}
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude']['units'] = "degrees_east"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude']['valid_min'] = -180.0
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude']['valid_max'] = 180.0
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['longitude']['coordinates'] = \
            "segment_id delta_time latitude"
        # segment ID
        IS2_atl06_mask[gtx]['land_ice_segments']['segment_id'] = segment_id
        IS2_atl06_fill[gtx]['land_ice_segments']['segment_id'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['segment_id'] = ['delta_time']
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['segment_id'] = {}
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['segment_id']['units'] = "1"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['segment_id']['contentType'] = "referenceInformation"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['segment_id']['long_name'] = "Along-track segment ID number"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['segment_id']['description'] = ("A 7 digit number "
            "identifying the along-track geolocation segment number.  These are sequential, starting with "
            "1 for the first segment after an ascending equatorial crossing node. Equal to the segment_id for "
            "the second of the two 20m ATL03 segments included in the 40m ATL06 segment")
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['segment_id']['coordinates'] = \
            "delta_time latitude longitude"

        # subsetting variables
        IS2_atl06_mask[gtx]['land_ice_segments']['subsetting'] = {}
        IS2_atl06_fill[gtx]['land_ice_segments']['subsetting'] = {}
        IS2_atl06_dims[gtx]['land_ice_segments']['subsetting'] = {}
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting'] = {}
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['Description'] = ("The subsetting group "
            "contains parameters used to reduce land ice segments to specific regions of interest.")
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['data_rate'] = ("Data within this group "
            "are stored at the land_ice_segments segment rate.")

        # output mask to HDF5
        IS2_atl06_mask[gtx]['land_ice_segments']['subsetting']['ice_gz'] = associated_map
        IS2_atl06_fill[gtx]['land_ice_segments']['subsetting']['ice_gz'] = None
        IS2_atl06_dims[gtx]['land_ice_segments']['subsetting']['ice_gz'] = ['delta_time']
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['ice_gz'] = {}
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['ice_gz']['contentType'] = "referenceInformation"
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['ice_gz']['long_name'] = 'Grounding Zone Mask'
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['ice_gz']['description'] = ("Grounding zone mask "
            "calculated using delineations from {0} buffered by {1:0.0f} km.".format(grounded_description[HEM],args.buffer))
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['ice_gz']['reference'] = grounded_reference[HEM]
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['ice_gz']['source'] = args.buffer
        IS2_atl06_mask_attrs[gtx]['land_ice_segments']['subsetting']['ice_gz']['coordinates'] = \
            "../segment_id ../delta_time ../latitude ../longitude"
        # wait for all processes to finish calculation
        comm.Barrier()

    # parallel h5py I/O does not support compression filters at this time
    if (comm.rank == 0) and valid_check:
        # output HDF5 files with output masks
        fargs=(PRD,'GROUNDING_ZONE_MASK',YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS,AUX)
        file_format = '{0}_{1}_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
        output_file = args.output_directory.joinpath(file_format.format(*fargs))
        # print file information
        logging.info(f'\t{str(output_file)}')
        # write to output HDF5 file
        HDF5_ATL06_mask_write(IS2_atl06_mask, IS2_atl06_mask_attrs,
            FILENAME=output_file,
            INPUT=GRANULE,
            FILL_VALUE=IS2_atl06_fill,
            DIMENSIONS=IS2_atl06_dims,
            CLOBBER=True)
        # change the permissions mode
        output_file.chmod(mode=args.mode)
    # close the input file
    fileID.close()

# PURPOSE: outputting the masks for ICESat-2 data to HDF5
def HDF5_ATL06_mask_write(IS2_atl06_mask, IS2_atl06_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=True):
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
    for k,v in IS2_atl06_mask['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl06_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam
    beams = [k for k in IS2_atl06_mask.keys() if bool(re.match(r'gt\d[lr]',k))]
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

        # segment_id, geolocation, time and height variables
        for k in ['delta_time','segment_id','latitude','longitude']:
            # values and attributes
            v = IS2_atl06_mask[gtx]['land_ice_segments'][k]
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

        # add to subsetting variables
        key = 'subsetting'
        fileID[gtx]['land_ice_segments'].create_group(key)
        h5[gtx]['land_ice_segments'][key] = {}
        for att_name in ['Description','data_rate']:
            att_val=IS2_atl06_attrs[gtx]['land_ice_segments'][key][att_name]
            fileID[gtx]['land_ice_segments'][key].attrs[att_name] = att_val
        for k,v in IS2_atl06_mask[gtx]['land_ice_segments'][key].items():
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
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Land Ice Height'
    fileID.attrs['summary'] = ('Subsetting masks for ice-sheets segments '
        'needed to interpret and assess the quality of land height estimates.')
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
        lon = IS2_atl06_mask[gtx]['land_ice_segments']['longitude']
        lat = IS2_atl06_mask[gtx]['land_ice_segments']['latitude']
        delta_time = IS2_atl06_mask[gtx]['land_ice_segments']['delta_time']
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
    fileID.attrs['software_reference'] = gz.version.project_name
    fileID.attrs['software_version'] = gz.version.full_version
    # Closing the HDF5 file
    fileID.close()

# run main program
if __name__ == '__main__':
    main()
