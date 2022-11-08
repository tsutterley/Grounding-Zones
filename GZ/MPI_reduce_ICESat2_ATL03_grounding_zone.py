#!/usr/bin/env python
u"""
MPI_reduce_ICESat2_ATL03_grounding_zone.py
Written by Tyler Sutterley (11/2022)

Create masks for reducing ICESat-2 geolocated photon height data to within
    a buffer region near the ice sheet grounding zone
Used to calculate a more definite grounding zone from the ICESat-2 data

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
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
    Updated 11/2022: verify coordinate reference system attribute from shapefile
    Updated 10/2022: simplied HDF5 file output to match other reduction programs
    Updated 08/2022: use logging for verbose output of processing run
    Updated 07/2022: place some imports within try/except statements
    Updated 05/2022: use argparse descriptions within documentation
    Updated 02/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 01/2021: time utilities for converting times from JD and to decimal
    Updated 12/2020: H5py deprecation warning change to use make_scale
    Updated 10/2020: using argparse to set parameters.  update pyproj transforms
    Updated 08/2020: using convert delta time function to convert to Julian days
    Updated 10/2019: using delta_time as output HDF5 variable dimensions
    Updated 09/2019: using fiona for shapefile read and pyproj for coordinates
    Updated 04/2019: check if subsetted beam contains land ice data
    Forked 04/2019 from MPI_reduce_triangulated_grounding_zone.py
    Updated 02/2019: shapely updates for python3 compatibility
    Updated 07/2017: using parts from shapefile
    Written 06/2017
"""
from __future__ import print_function

import os
import re
import h5py
import pyproj
import logging
import argparse
import datetime
import warnings
import numpy as np
from grounding_zones.utilities import get_data_path
from icesat2_toolkit.convert_delta_time import convert_delta_time
import icesat2_toolkit.time
#-- attempt imports
try:
    import fiona
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("fiona not available")
try:
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("mpi4py not available")
try:
    import shapely.geometry
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("shapely not available")
#-- ignore warnings
warnings.filterwarnings("ignore")

#-- buffered shapefile
buffer_shapefile = {}
buffer_shapefile['N'] = 'grn_ice_sheet_buffer_{0:0.0f}km.shp'
buffer_shapefile['S'] = 'ant_ice_sheet_islands_v2_buffer_{0:0.0f}km.shp'
#-- description and reference for each grounded ice file
grounded_description = {}
grounded_description['N'] = 'Greenland Mapping Project (GIMP) Ice & Ocean Mask'
grounded_description['S'] = ('MEaSUREs Antarctic Boundaries for IPY 2007-2009 '
    'from Satellite_Radar, Version 2')
grounded_reference = {}
grounded_reference['N'] = 'https://doi.org/10.5194/tc-8-1509-2014'
grounded_reference['S'] = 'https://doi.org/10.5067/IKBWW4RYHF1Q'

#-- PURPOSE: keep track of MPI threads
def info(rank, size):
    logging.info('Rank {0:d} of {1:d}'.format(rank+1,size))
    logging.info('module name: {0}'.format(__name__))
    if hasattr(os, 'getppid'):
        logging.info('parent process: {0:d}'.format(os.getppid()))
    logging.info('process id: {0:d}'.format(os.getpid()))

#-- PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Create masks for reducing ICESat-2 geolocated
            photon height data to within a buffer region near the
            ice sheet grounding zone
            """
    )
    #-- command line parameters
    parser.add_argument('file',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='ICESat-2 ATL03 file to run')
    #-- working data directory for shapefiles
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=get_data_path('data'),
        help='Working data directory')
    #-- buffer in kilometers for extracting grounding zone
    parser.add_argument('--buffer','-B',
        type=float, default=20.0,
        help='Distance in kilometers to buffer grounding zone')
    #-- verbosity settings
    #-- verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    #-- permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output files')
    #-- return the parser
    return parser

#-- PURPOSE: set the hemisphere of interest based on the granule
def set_hemisphere(GRANULE):
    if GRANULE in ('10','11','12'):
        projection_flag = 'S'
    elif GRANULE in ('03','04','05'):
        projection_flag = 'N'
    return projection_flag

#-- PURPOSE: load the polygon object for the buffered estimated grounding zone
def load_grounding_zone(base_dir, HEM, BUFFER):
    #-- buffered shapefile for region
    buffered_shapefile = buffer_shapefile[HEM].format(BUFFER)
    logging.info(os.path.join(base_dir,buffered_shapefile))
    #-- read buffered shapefile
    shape_input = fiona.open(os.path.join(base_dir,buffered_shapefile))
    #-- extract coordinate reference system
    if ('init' in shape_input.crs.keys()):
        epsg = pyproj.CRS(shape_input.crs['init']).to_epsg()
    else:
        epsg = pyproj.CRS(shape_input.crs).to_epsg()
    #-- create list of polygons
    polygons = []
    #-- extract the entities and assign by tile name
    for i,ent in enumerate(shape_input.values()):
        #-- list of coordinates
        poly_list = []
        #-- extract coordinates for entity
        for coords in ent['geometry']['coordinates']:
            #-- extract Polar-Stereographic coordinates for record
            x,y = np.transpose(coords)
            poly_list.append(list(zip(x,y)))
        #-- convert poly_list into Polygon object with holes
        poly_obj = shapely.geometry.Polygon(poly_list[0],holes=poly_list[1:])
        #-- Valid Polygon cannot have overlapping exterior or interior rings
        if (not poly_obj.is_valid):
            poly_obj = poly_obj.buffer(0)
        polygons.append(poly_obj)
    #-- create shapely multipolygon object
    mpoly_obj = shapely.geometry.MultiPolygon(polygons)
    #-- close the shapefile
    shape_input.close()
    #-- return the polygon object for the ice sheet
    return (mpoly_obj,buffered_shapefile,epsg)

#-- PURPOSE: read ICESat-2 geolocated photon height data (ATL03)
#-- reduce data to within buffer of grounding zone
def main():
    #-- start MPI communicator
    comm = MPI.COMM_WORLD

    #-- Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    #-- create logger
    loglevel = logging.INFO if args.verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    #-- output module information for process
    info(comm.rank,comm.size)
    if (comm.rank == 0):
        logging.info('{0} -->'.format(args.file))

    #-- Open the HDF5 file for reading
    fileID = h5py.File(args.file, 'r', driver='mpio', comm=comm)
    DIRECTORY = os.path.dirname(args.file)
    #-- extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS,AUX=rx.findall(args.file).pop()
    #-- set the hemisphere flag based on ICESat-2 granule
    HEM = set_hemisphere(GRN)
    #-- read each input beam within the file
    IS2_atl03_beams = []
    for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]',k))]:
        #-- check if subsetted beam contains data
        #-- check in both the geolocation and heights groups
        try:
            fileID[gtx]['geolocation']['segment_id']
            fileID[gtx]['heights']['delta_time']
        except KeyError:
            pass
        else:
            IS2_atl03_beams.append(gtx)

    #-- read data on rank 0
    if (comm.rank == 0):
        #-- read shapefile and create shapely multipolygon objects
        mpoly_obj,_,epsg = load_grounding_zone(args.directory,HEM,args.buffer)
    else:
        #-- create empty object for list of shapely objects
        mpoly_obj = None
        epsg = None

    #-- Broadcast Shapely multipolygon objects and projection
    mpoly_obj = comm.bcast(mpoly_obj, root=0)
    epsg = comm.bcast(epsg, root=0)

    #-- pyproj transformer for converting lat/lon to polar stereographic
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    crs2 = pyproj.CRS.from_epsg(epsg)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    #-- copy variables for outputting to HDF5 file
    IS2_atl03_mask = {}
    IS2_atl03_fill = {}
    IS2_atl03_dims = {}
    IS2_atl03_mask_attrs = {}
    #-- combined validity check for all beams
    valid_check = False
    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    #-- Add this value to delta time parameters to compute full gps_seconds
    IS2_atl03_mask['ancillary_data'] = {}
    IS2_atl03_mask_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        #-- get each HDF5 variable
        IS2_atl03_mask['ancillary_data'][key] = fileID['ancillary_data'][key][:]
        #-- Getting attributes of group and included variables
        IS2_atl03_mask_attrs['ancillary_data'][key] = {}
        for att_name,att_val in fileID['ancillary_data'][key].attrs.items():
            IS2_atl03_mask_attrs['ancillary_data'][key][att_name] = att_val

    #-- for each input beam within the file
    for gtx in sorted(IS2_atl03_beams):
        #-- output data dictionaries for beam
        IS2_atl03_mask[gtx] = dict(heights={},subsetting={})
        IS2_atl03_fill[gtx] = dict(heights={},subsetting={})
        IS2_atl03_dims[gtx] = dict(heights={},subsetting={})
        IS2_atl03_mask_attrs[gtx] = dict(heights={},subsetting={})

        #-- number of photon events
        n_pe, = fileID[gtx]['heights']['h_ph'].shape
        #-- check if there are less photon events than processes
        if (n_pe < comm.Get_size()):
            continue
        #-- define indices to run for specific process
        ind = np.arange(comm.Get_rank(), n_pe, comm.Get_size(), dtype=int)

        #-- extract delta time
        delta_time = fileID[gtx]['heights']['delta_time'][:]
        #-- extract lat/lon
        longitude = fileID[gtx]['heights']['lon_ph'][:]
        latitude = fileID[gtx]['heights']['lat_ph'][:]

        #-- convert lat/lon to polar stereographic
        X,Y = transformer.transform(longitude[ind], latitude[ind])
        #-- convert reduced x and y to shapely multipoint object
        xy_point = shapely.geometry.MultiPoint(np.c_[X, Y])

        #-- create distributed intersection map for calculation
        distributed_map = np.zeros((n_pe),dtype=bool)
        #-- create empty intersection map array for receiving
        associated_map = np.zeros((n_pe),dtype=bool)
        #-- for each polygon
        for poly_obj in mpoly_obj:
            #-- finds if points are encapsulated (in grounding zone)
            int_test = poly_obj.intersects(xy_point)
            if int_test:
                #-- extract intersected points
                int_map = list(map(poly_obj.intersects,xy_point))
                int_indices, = np.nonzero(int_map)
                #-- set distributed_map indices to True for intersected points
                distributed_map[ind[int_indices]] = True
        #-- communicate output MPI matrices between ranks
        #-- operation is a logical "or" across the elements.
        comm.Allreduce(sendbuf=[distributed_map, MPI.BOOL], \
            recvbuf=[associated_map, MPI.BOOL], op=MPI.LOR)
        distributed_map = None
        #-- wait for all processes to finish calculation
        comm.Barrier()
        #-- add to validity check
        valid_check |= np.any(associated_map)

        #-- group attributes for beam
        IS2_atl03_mask_attrs[gtx]['Description'] = fileID[gtx].attrs['Description']
        IS2_atl03_mask_attrs[gtx]['atlas_pce'] = fileID[gtx].attrs['atlas_pce']
        IS2_atl03_mask_attrs[gtx]['atlas_beam_type'] = fileID[gtx].attrs['atlas_beam_type']
        IS2_atl03_mask_attrs[gtx]['groundtrack_id'] = fileID[gtx].attrs['groundtrack_id']
        IS2_atl03_mask_attrs[gtx]['atmosphere_profile'] = fileID[gtx].attrs['atmosphere_profile']
        IS2_atl03_mask_attrs[gtx]['atlas_spot_number'] = fileID[gtx].attrs['atlas_spot_number']
        IS2_atl03_mask_attrs[gtx]['sc_orientation'] = fileID[gtx].attrs['sc_orientation']
        #-- group attributes for heights
        IS2_atl03_mask_attrs[gtx]['heights']['Description'] = ("Contains arrays of the "
            "parameters for each received photon.")
        IS2_atl03_mask_attrs[gtx]['heights']['data_rate'] = ("Data are stored at the "
            "photon detection rate.")

        #-- geolocation, time and segment ID
        #-- delta time
        IS2_atl03_mask[gtx]['heights']['delta_time'] = delta_time
        IS2_atl03_fill[gtx]['heights']['delta_time'] = None
        IS2_atl03_dims[gtx]['heights']['delta_time'] = None
        IS2_atl03_mask_attrs[gtx]['heights']['delta_time'] = {}
        IS2_atl03_mask_attrs[gtx]['heights']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl03_mask_attrs[gtx]['heights']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl03_mask_attrs[gtx]['heights']['delta_time']['standard_name'] = "time"
        IS2_atl03_mask_attrs[gtx]['heights']['delta_time']['calendar'] = "standard"
        IS2_atl03_mask_attrs[gtx]['heights']['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl03_mask_attrs[gtx]['heights']['delta_time']['coordinates'] = \
            "lat_ph lon_ph"
        #-- latitude
        IS2_atl03_mask[gtx]['heights']['latitude'] = latitude
        IS2_atl03_fill[gtx]['heights']['latitude'] = None
        IS2_atl03_dims[gtx]['heights']['latitude'] = ['delta_time']
        IS2_atl03_mask_attrs[gtx]['heights']['latitude'] = {}
        IS2_atl03_mask_attrs[gtx]['heights']['latitude']['units'] = "degrees_north"
        IS2_atl03_mask_attrs[gtx]['heights']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl03_mask_attrs[gtx]['heights']['latitude']['long_name'] = "Latitude"
        IS2_atl03_mask_attrs[gtx]['heights']['latitude']['standard_name'] = "latitude"
        IS2_atl03_mask_attrs[gtx]['heights']['latitude']['description'] = ("Latitude of each "
            "received photon. Computed from the ECF Cartesian coordinates of the bounce point.")
        IS2_atl03_mask_attrs[gtx]['heights']['latitude']['valid_min'] = -90.0
        IS2_atl03_mask_attrs[gtx]['heights']['latitude']['valid_max'] = 90.0
        IS2_atl03_mask_attrs[gtx]['heights']['latitude']['coordinates'] = \
            "delta_time lon_ph"
        #-- longitude
        IS2_atl03_mask[gtx]['heights']['longitude'] = longitude
        IS2_atl03_fill[gtx]['heights']['longitude'] = None
        IS2_atl03_dims[gtx]['heights']['longitude'] = ['delta_time']
        IS2_atl03_mask_attrs[gtx]['heights']['longitude'] = {}
        IS2_atl03_mask_attrs[gtx]['heights']['longitude']['units'] = "degrees_east"
        IS2_atl03_mask_attrs[gtx]['heights']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl03_mask_attrs[gtx]['heights']['longitude']['long_name'] = "Longitude"
        IS2_atl03_mask_attrs[gtx]['heights']['longitude']['standard_name'] = "longitude"
        IS2_atl03_mask_attrs[gtx]['heights']['longitude']['description'] = ("Longitude of each "
            "received photon. Computed from the ECF Cartesian coordinates of the bounce point.")
        IS2_atl03_mask_attrs[gtx]['heights']['longitude']['valid_min'] = -180.0
        IS2_atl03_mask_attrs[gtx]['heights']['longitude']['valid_max'] = 180.0
        IS2_atl03_mask_attrs[gtx]['heights']['longitude']['coordinates'] = \
            "delta_time lat_ph"

        #-- subsetting variables
        IS2_atl03_mask_attrs[gtx]['subsetting']['Description'] = ("The subsetting group "
            "contains parameters used to reduce photon events to specific regions of interest.")
        IS2_atl03_mask_attrs[gtx]['subsetting']['data_rate'] = ("Data are stored at the photon "
            "detection rate.")

        #-- output mask to HDF5
        IS2_atl03_mask[gtx]['subsetting']['ice_gz'] = associated_map
        IS2_atl03_fill[gtx]['subsetting']['ice_gz'] = None
        IS2_atl03_dims[gtx]['subsetting']['ice_gz'] = ['delta_time']
        IS2_atl03_mask_attrs[gtx]['subsetting']['ice_gz'] = {}
        IS2_atl03_mask_attrs[gtx]['subsetting']['ice_gz']['contentType'] = "referenceInformation"
        IS2_atl03_mask_attrs[gtx]['subsetting']['ice_gz']['long_name'] = 'Grounding Zone Mask'
        IS2_atl03_mask_attrs[gtx]['subsetting']['ice_gz']['description'] = ("Grounding zone mask "
            "calculated using delineations from {0} buffered by {1:0.0f} km.".format(
            grounded_description[HEM],args.buffer))
        IS2_atl03_mask_attrs[gtx]['subsetting']['ice_gz']['reference'] = grounded_reference[HEM]
        IS2_atl03_mask_attrs[gtx]['subsetting']['ice_gz']['source'] = args.buffer
        IS2_atl03_mask_attrs[gtx]['subsetting']['ice_gz']['coordinates'] = \
            "../heights/delta_time ../heights/lat_ph ../heights/lon_ph"
        #-- wait for all processes to finish calculation
        comm.Barrier()

    #-- parallel h5py I/O does not support compression filters at this time
    if (comm.rank == 0) and valid_check:
        #-- output HDF5 files with output masks
        fargs=(PRD,'GROUNDING_ZONE_MASK',YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS,AUX)
        file_format='{0}_{1}_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
        output_file=os.path.join(DIRECTORY,file_format.format(*fargs))
        #-- print file information
        logging.info('\t{0}'.format(output_file))
        #-- write to output HDF5 file
        HDF5_ATL03_mask_write(IS2_atl03_mask, IS2_atl03_mask_attrs,
            CLOBBER=True, INPUT=os.path.basename(args.file),
            FILL_VALUE=IS2_atl03_fill, DIMENSIONS=IS2_atl03_dims,
            FILENAME=output_file)
        #-- change the permissions mode
        os.chmod(output_file, args.mode)
    #-- close the input file
    fileID.close()

#-- PURPOSE: outputting the masks for ICESat-2 data to HDF5
def HDF5_ATL03_mask_write(IS2_atl03_mask, IS2_atl03_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=True):
    #-- setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    #-- open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    #-- create HDF5 records
    h5 = {}

    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl03_mask['ancillary_data'].items():
        #-- Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        #-- add HDF5 variable attributes
        for att_name,att_val in IS2_atl03_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    #-- write each output beam
    beams = [k for k in IS2_atl03_mask.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        #-- add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl03_attrs[gtx][att_name]

        #-- for each output data group
        for key in ['heights','subsetting']:
            #-- create group
            fileID[gtx].create_group(key)
            h5[gtx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val = IS2_atl03_attrs[gtx][key][att_name]
                fileID[gtx][key].attrs[att_name] = att_val

            #-- all variables for group
            groupkeys=set(IS2_atl03_mask[gtx][key].keys())-set(['delta_time'])
            for k in ['delta_time',*sorted(groupkeys)]:
                #-- values and attributes
                v = IS2_atl03_mask[gtx][key][k]
                attrs = IS2_atl03_attrs[gtx][key][k]
                fillvalue = FILL_VALUE[gtx][key][k]
                #-- Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(gtx,key,k)
                if fillvalue:
                    h5[gtx][key][k] = fileID.create_dataset(val,
                        np.shape(v), data=v, dtype=v.dtype,
                        fillvalue=fillvalue, compression='gzip')
                else:
                    h5[gtx][key][k] = fileID.create_dataset(val,
                        np.shape(v), data=v, dtype=v.dtype,
                        compression='gzip')
                #-- create or attach dimensions for HDF5 variable
                if DIMENSIONS[gtx][key][k]:
                    #-- attach dimensions
                    for i,dim in enumerate(DIMENSIONS[gtx][key][k]):
                        h5[gtx][key][k].dims[i].attach_scale(
                            h5[gtx][key][dim])
                else:
                    #-- make dimension
                    h5[gtx][key][k].make_scale(k)
                #-- add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[gtx][key][k].attrs[att_name] = att_val

    #-- HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L2A Global Geolocated Photon Data'
    fileID.attrs['summary'] = ("The purpose of ATL03 is to provide along-track "
        "photon data for all 6 ATLAS beams and associated statistics.")
    fileID.attrs['description'] = ("Photon heights determined by ATBD "
        "Algorithm using POD and PPD. All photon events per transmit pulse per "
        "beam. Includes POD and PPD vectors. Classification of each photon by "
        "several ATBD Algorithms.")
    date_created = datetime.datetime.today()
    fileID.attrs['date_created'] = date_created.isoformat()
    project = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = project
    platform = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = platform
    #-- add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    #-- add attributes for input ATL03 and ATL09 files
    fileID.attrs['input_files'] = ','.join([os.path.basename(i) for i in INPUT])
    #-- find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl03_mask[gtx]['heights']['longitude']
        lat = IS2_atl03_mask[gtx]['heights']['latitude']
        delta_time = IS2_atl03_mask[gtx]['heights']['delta_time']
        #-- setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time.min() if (delta_time.min() < tmn) else tmn
        tmx = delta_time.max() if (delta_time.max() > tmx) else tmx
    #-- add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = ltmn
    fileID.attrs['geospatial_lat_max'] = ltmx
    fileID.attrs['geospatial_lon_min'] = lnmn
    fileID.attrs['geospatial_lon_max'] = lnmx
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['date_type'] = 'UTC'
    fileID.attrs['time_type'] = 'CCSDS UTC-A'
    #-- convert start and end time from ATLAS SDP seconds into UTC time
    time_utc = convert_delta_time(np.array([tmn,tmx]))
    #-- convert to calendar date
    YY,MM,DD,HH,MN,SS = icesat2_toolkit.time.convert_julian(time_utc['julian'],
        FORMAT='tuple')
    #-- add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = '{0:0.0f}'.format(tmx-tmn)
    #-- Closing the HDF5 file
    fileID.close()

#-- run main program
if __name__ == '__main__':
    main()
