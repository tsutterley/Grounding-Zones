#!/usr/bin/env python
u"""
MPI_interpolate_DEM.py
Written by Tyler Sutterley (09/2019)
Determines which digital elevation model tiles to read for set of coordinates
Reads 3x3 array of tiles for points within bounding box of central mosaic tile
Interpolates digital elevation model to coordinates

ArcticDEM 2m digital elevation model tiles
	http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/
	http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/

REMA 8m digital elevation model tiles
	http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/
	http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/

GIMP 30m digital elevation model tiles computed with nsidc_convert_GIMP_DEM.py
	https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0645.001/

COMMAND LINE OPTIONS:
	-D X, --directory=X: Working data directory
	--model=X: Set the digital elevation model (REMA, ArcticDEM, GIMP) to run
	-M X, --mode=X: Permission mode of directories and files created
	-V, --verbose: Output information about each created file

REQUIRES MPI PROGRAM
	MPI: standardized and portable message-passing system
		https://www.open-mpi.org/
		http://mpitutorial.com/

PYTHON DEPENDENCIES:
	numpy: Scientific Computing Tools For Python
		http://www.numpy.org
		http://www.scipy.org/NumPy_for_Matlab_Users
	scipy: Scientific Tools for Python
		http://www.scipy.org/
	mpi4py: MPI for Python
		http://pythonhosted.org/mpi4py/
		http://mpi4py.readthedocs.org/en/stable/
	h5py: Python interface for Hierarchal Data Format 5 (HDF5)
		http://h5py.org
		http://docs.h5py.org/en/stable/mpi.html
	fiona: Python wrapper for vector data access functions from the OGR library
		https://fiona.readthedocs.io/en/latest/manual.html
	gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
		https://pypi.python.org/pypi/GDAL/
	shapely: PostGIS-ish operations outside a database context for Python
		http://toblerity.org/shapely/index.html
	pyproj: Python interface to PROJ library
		https://pypi.org/project/pyproj/

REFERENCES:
	https://www.pgc.umn.edu/guides/arcticdem/data-description/
	https://www.pgc.umn.edu/guides/rema/data-description/
	https://nsidc.org/data/nsidc-0645/versions/1

UPDATE HISTORY:
	Updated 09/2019: round fill value for mask as some tiles can be incorrect
	Forked 09/2019 from MPI_DEM_ICESat2_ATL03.py
"""
from __future__ import print_function

import sys
import os
import re
import uuid
import fiona
import getopt
import pyproj
import tarfile
import datetime
import osgeo.gdal
import numpy as np
from mpi4py import MPI
import scipy.interpolate
from shapely.geometry import MultiPoint, Polygon

#-- digital elevation models
elevation_dir = {}
elevation_tile_index = {}
#-- ArcticDEM
elevation_dir['ArcticDEM'] = ['ArcticDEM']
elevation_tile_index['ArcticDEM'] = 'ArcticDEM_Tile_Index_Rel7.zip'
#-- GIMP DEM
elevation_dir['GIMP'] = ['GIMP','30m']
elevation_tile_index['GIMP'] = 'gimpdem_Tile_Index_Rel1.1.zip'
#-- REMA DEM
elevation_dir['REMA'] = ['REMA']
elevation_tile_index['REMA'] = 'REMA_Tile_Index_Rel1.1.zip'

#-- PURPOSE: help module to describe the optional input parameters
def usage():
	print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
	print(' -D X, --directory=X\tWorking data directory')
	print(' -M X, --mode=X\t\tPermission mode of directories and files created')
	print(' -V, --verbose\t\tOutput information about each created file\n')

#-- PURPOSE: keep track of MPI threads
def info(rank, size):
	print('Rank {0:d} of {1:d}'.format(rank+1,size))
	print('module name: {0}'.format(__name__))
	if hasattr(os, 'getppid'):
		print('parent process: {0:d}'.format(os.getppid()))
	print('process id: {0:d}'.format(os.getpid()))

#-- PURPOSE: reading the number of file lines removing commented lines
def file_length(input_file):
	#-- read the input file, split at lines and remove all commented lines
	with open(input_file,'r') as f:
		i = [i for i in f.read().splitlines() if re.match('^(?!#)',i)]
	#-- return the number of lines
	return len(i)

#-- PURPOSE: read zip file containing index shapefiles for finding DEM tiles
def read_DEM_index(index_file, DEM_MODEL):
	#-- read the compressed shapefile and extract entities
	shape = fiona.open('zip://{0}'.format(os.path.expanduser(index_file)))
	epsg = shape.crs['init']
	#-- extract attribute indice for DEM tile (REMA,GIMP) or name (ArcticDEM)
	if (DEM_MODEL == 'REMA'):
		#-- REMA index file attributes:
		#-- name: DEM mosaic name for tile (file name without suffix)
		#-- tile: DEM tile identifier (IMy_IMx)
		#-- nd_value: fill value for elements with no data
		#-- resolution: DEM horizontal spatial resolution (meters)
		#-- creationda: creation date
		#-- raster: (empty)
		#-- fileurl: link to file on PGC server
		#-- spec_type: specific type (DEM)
		#-- qual: density of scenes within tile (0 to 1)
		#-- reg_src: DEM registration source (ICESat or neighbor align)
		#-- num_gcps: number of ground control points
		#-- meanresz: mean vertical residual (meters)
		#-- active: (1)
		#-- qc: (2)
		#-- rel_ver: release version
		#-- num_comp: number of components
		#-- st_area_sh: tile area (meters^2)
		#-- st_length_: perimeter length of tile (meters)
		field = 'tile'
	elif (DEM_MODEL == 'GIMP'):
		#-- GIMP index file attributes (from make_GIMP_tile_shapefile.py):
		#-- name: DEM mosaic name for tile (file name without suffix)
		#-- tile: DEM tile identifier (IMy_IMx)
		#-- nd_value: fill value for elements with no data
		#-- resolution: DEM horizontal spatial resolution (meters)
		#-- fileurl: link to file on NSIDC server
		#-- spec_type: specific type (DEM)
		#-- reg_src: DEM registration source (ICESat or neighbor align)
		#-- rel_ver: release version
		#-- num_comp: number of components
		#-- st_area_sh: tile area (meters^2)
		#-- st_length_: perimeter length of tile (meters)
		field = 'tile'
	elif (DEM_MODEL == 'ArcticDEM'):
		#-- ArcticDEM index file attributes:
		#-- objectid: DEM tile object identifier for sub-tile
		#-- name: DEM mosaic name for sub-tile (file name without suffix)
		#-- tile: DEM tile identifier (IMy_IMx) (non-unique for sub-tiles)
		#-- nd_value: fill value for elements with no data
		#-- resolution: DEM horizontal spatial resolution (meters)
		#-- creationda: creation date
		#-- raster: (empty)
		#-- fileurl: link to file on PGC server
		#-- spec_type: specific type (DEM)
		#-- qual: density of scenes within tile (0 to 1)
		#-- reg_src: DEM registration source (ICESat or neighbor align)
		#-- num_gcps: number of ground control points
		#-- meanresz: mean vertical residual (meters)
		#-- active: (1)
		#-- qc: (2)
		#-- rel_ver: release version
		#-- num_comp: number of components
		#-- st_area_sh: tile area (meters^2)
		#-- st_length_: perimeter length of tile (meters)
		field = 'name'
	#-- create python dictionary for each polygon object
	poly_dict = {}
	attrs_dict = {}
	#-- extract the entities and assign by tile name
	for i,ent in enumerate(shape.values()):
		#-- tile or name attributes
		if DEM_MODEL in ('REMA','GIMP'):
			tile = str(ent['properties'][field])
		else:
			tile, = re.findall(r'^(\d+_\d+_\d+_\d+)',ent['properties'][field])
		#-- extract attributes and assign by tile
		attrs_dict[tile] = {}
		for key,val in ent['properties'].items():
			attrs_dict[tile][key] = val
		#-- upper-left, upper-right, lower-right, lower-left, upper-left
		ul,ur,lr,ll,ul2 = ent['geometry']['coordinates'].pop()
		#-- tile boundaries
		attrs_dict[tile]['xmin'] = ul[0]
		attrs_dict[tile]['xmax'] = lr[0]
		attrs_dict[tile]['ymin'] = lr[1]
		attrs_dict[tile]['ymax'] = ul[1]
		#-- extract Polar Stereographic coordinates for entity
		x = [ul[0],ur[0],lr[0],ll[0],ul2[0]]
		y = [ul[1],ur[1],lr[1],ll[1],ul2[1]]
		poly_obj = Polygon(list(zip(x,y)))
		#-- Valid Polygon may not possess overlapping exterior or interior rings
		if (not poly_obj.is_valid):
			poly_obj = poly_obj.buffer(0)
		poly_dict[tile] = poly_obj
	#-- close the file
	shape.close()
	#-- return the dictionaries of polygon objects and attributes
	return (poly_dict,attrs_dict,epsg)

#-- PURPOSE: read DEM tile file from gzipped tar files
def read_DEM_file(elevation_file):
	#-- open file with tarfile (read)
	tar = tarfile.open(name=elevation_file, mode='r:gz')
	#-- find dem geotiff file within tar file
	member, = [m for m in tar.getmembers() if re.search('dem\.tif',m.name)]
	#-- use GDAL memory-mapped file to read dem
	mmap_name = "/vsimem/{0}".format(uuid.uuid4().hex)
	osgeo.gdal.FileFromMemBuffer(mmap_name, tar.extractfile(member).read())
	ds = osgeo.gdal.Open(mmap_name)
	#-- read data matrix
	im = ds.GetRasterBand(1).ReadAsArray()
	fill_value = ds.GetRasterBand(1).GetNoDataValue()
	fill_value = 0.0 if (fill_value is None) else fill_value
	#-- get dimensions
	xsize = ds.RasterXSize
	ysize = ds.RasterYSize
	#-- create mask for finding invalid values
	mask = np.zeros((ysize,xsize),dtype=np.bool)
	indy,indx = np.nonzero((im == fill_value) | (~np.isfinite(im)) |
		(np.ceil(im) == np.ceil(fill_value)))
	mask[indy,indx] = True
	#-- verify that values are finite by replacing with fill_value
	im[indy,indx] = fill_value
	#-- get geotiff info
	info_geotiff = ds.GetGeoTransform()
	#-- calculate image extents
	xmin = info_geotiff[0]
	ymax = info_geotiff[3]
	xmax = xmin + (xsize-1)*info_geotiff[1]
	ymin = ymax + (ysize-1)*info_geotiff[5]
	#-- close files
	ds = None
	osgeo.gdal.Unlink(mmap_name)
	tar.close()
	#-- create image x and y arrays
	xi = np.arange(xmin,xmax+info_geotiff[1],info_geotiff[1])
	yi = np.arange(ymax,ymin+info_geotiff[5],info_geotiff[5])
	#-- return values (flip y values to be monotonically increasing)
	return (im[::-1,:],mask[::-1,:],fill_value,xi,yi[::-1])

#-- PURPOSE: read DEM tile file from gzipped tar files to buffer main tile
def read_DEM_buffer(elevation_file, xlimits, ylimits):
	#-- open file with tarfile (read)
	tar = tarfile.open(name=elevation_file, mode='r:gz')
	#-- find dem geotiff file within tar file
	member, = [m for m in tar.getmembers() if re.search('dem\.tif',m.name)]
	#-- use GDAL memory-mapped file to read dem
	mmap_name = "/vsimem/{0}".format(uuid.uuid4().hex)
	osgeo.gdal.FileFromMemBuffer(mmap_name, tar.extractfile(member).read())
	ds = osgeo.gdal.Open(mmap_name)
	#-- get geotiff info
	info_geotiff = ds.GetGeoTransform()
	#-- original image extents
	xmin = info_geotiff[0]
	ymax = info_geotiff[3]
	#-- reduce input image with GDAL
	#-- Specify offset and rows and columns to read
	xoffset = np.int((xlimits[0] - xmin)/info_geotiff[1])
	yoffset = np.int((ymax - ylimits[1])/np.abs(info_geotiff[5]))
	xcount = np.int((xlimits[1] - xlimits[0])/info_geotiff[1]) + 1
	ycount = np.int((ylimits[1] - ylimits[0])/np.abs(info_geotiff[5])) + 1
	#-- read data matrix
	im = ds.GetRasterBand(1).ReadAsArray(xoffset, yoffset, xcount, ycount)
	fill_value = ds.GetRasterBand(1).GetNoDataValue()
	fill_value = 0.0 if (fill_value is None) else fill_value
	#-- create mask for finding invalid values
	mask = np.zeros((ycount,xcount))
	indy,indx = np.nonzero((im == fill_value) | (~np.isfinite(im)) |
		(np.ceil(im) == np.ceil(fill_value)))
	mask[indy,indx] = True
	#-- verify that values are finite by replacing with fill_value
	im[indy,indx] = fill_value
	#-- reduced x and y limits of image
	xmin_reduced = xmin + xoffset*info_geotiff[1]
	xmax_reduced = xmin + xoffset*info_geotiff[1] + (xcount-1)*info_geotiff[1]
	ymax_reduced = ymax + yoffset*info_geotiff[5]
	ymin_reduced = ymax + yoffset*info_geotiff[5] + (ycount-1)*info_geotiff[5]
	#-- close files
	ds = None
	osgeo.gdal.Unlink(mmap_name)
	tar.close()
	#-- create image x and y arrays
	xi = np.arange(xmin_reduced,xmax_reduced+info_geotiff[1],info_geotiff[1])
	yi = np.arange(ymax_reduced,ymin_reduced+info_geotiff[5],info_geotiff[5])
	#-- return values (flip y values to be monotonically increasing)
	return (im[::-1,:],mask[::-1,:],fill_value,xi,yi[::-1])

#-- PURPOSE: read csv file of lat,lon coordinates
#-- convert latitude and longitude to polar stereographic
#-- interpolate DEM data to x and y coordinates
def main():
	#-- start MPI communicator
	comm = MPI.COMM_WORLD

	#-- Read the system arguments listed after the program
	long_options=['help','directory=','model=','mode=','verbose']
	optlist,arglist = getopt.getopt(sys.argv[1:], 'hD:M:V', long_options)

	#-- working data directory
	base_dir = os.getcwd()
	#-- set the DEM model to run
	DEM_MODEL = None
	#-- verbosity settings
	VERBOSE = False
	#-- permissions mode of the local files (number in octal)
	MODE = 0o775
	for opt, arg in optlist:
		if opt in ('-h','--help'):
			usage()
			sys.exit()
		elif opt in ("-D","--directory"):
			base_dir = os.path.expanduser(arg)
		elif opt in ("--model"):
			DEM_MODEL = arg
		elif opt in ("-V","--verbose"):
			#-- output module information for process
			info(comm.rank,comm.size)
			VERBOSE = True
		elif opt in ("-M","--mode"):
			MODE = int(arg, 8)

	#-- enter HDF5 file as system argument
	if not arglist:
		raise IOError('No input file entered as system arguments')
	#-- tilde-expansion of listed input file
	FILE = os.path.expanduser(arglist[0])

	#-- read data from input file
	print('{0} -->'.format(os.path.basename(FILE))) if VERBOSE else None
	#-- number of data points
	n_pts = file_length(FILE)
	#-- define indices to run for specific process
	ind = np.arange(comm.Get_rank(), n_pts, comm.Get_size(), dtype=np.int)

	#-- regular expression pattern for extracting parameters from ArcticDEM name
	rx1 = re.compile('(\d+)_(\d+)_(\d+)_(\d+)_(\d+m)_(.*?)$', re.VERBOSE)
	#-- full path to DEM directory
	elevation_directory=os.path.join(base_dir,*elevation_dir[DEM_MODEL])
	#-- zip file containing index shapefiles for finding DEM tiles
	index_file=os.path.join(elevation_directory,elevation_tile_index[DEM_MODEL])

	#-- read data on rank 0
	if (comm.rank == 0):
		#-- read input data file (lat,lon,height)
		dinput = np.loadtxt(FILE,delimiter=',')
		#-- extract lat/lon
		latitude = dinput[:,0]
		longitude = dinput[:,1]
		#-- read index file for determining which tiles to read
		tile_dict,tile_attrs,tile_epsg = read_DEM_index(index_file,DEM_MODEL)
		#-- convert tile projection from latitude longitude to tile EPSG
		proj1 = pyproj.Proj("+init=EPSG:{0:d}".format(4326))
		proj2 = pyproj.Proj("+init={0}".format(tile_epsg))
		X,Y = pyproj.transform(proj1, proj2, longitude, latitude)
	else:
		#-- empty numpy arrays for x and y coordinates
		X = np.empty((n_pts))
		Y = np.empty((n_pts))
		#-- create empty object for list of shapely objects
		tile_dict = None
		tile_attrs = None

	#-- Broadcast x and y coordinates from rank 0 to all other ranks
	comm.Bcast([X, MPI.DOUBLE])
	comm.Bcast([Y, MPI.DOUBLE])
	#-- Broadcast Shapely polygon objects
	tile_dict = comm.bcast(tile_dict, root=0)
	tile_attrs = comm.bcast(tile_attrs, root=0)

	#-- output interpolated digital elevation model
	distributed_dem = np.ma.zeros((n_pts),fill_value=-9999.0,dtype=np.float32)
	distributed_dem.mask = np.ones((n_pts),dtype=np.bool)
	dem_h = np.ma.zeros((n_pts),fill_value=-9999.0,dtype=np.float32)
	dem_h.mask = np.ones((n_pts),dtype=np.bool)

	#-- convert reduced x and y to shapely multipoint object
	xy_point = MultiPoint(list(zip(X[ind], Y[ind])))

	#-- create complete masks for each DEM tile
	associated_map = {}
	for key,poly_obj in tile_dict.items():
		#-- create empty intersection map array for distributing
		distributed_map = np.zeros((n_pts),dtype=np.int)
		#-- create empty intersection map array for receiving
		associated_map[key] = np.zeros((n_pts),dtype=np.int)
		#-- finds if points are encapsulated (within tile)
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
			recvbuf=[associated_map[key], MPI.BOOL], op=MPI.LOR)
		distributed_map = None
	#-- wait for all processes to finish calculation
	comm.Barrier()
	#-- find valid tiles and free up memory from invalid tiles
	valid_tiles = [k for k,v in associated_map.items() if v.any()]
	invalid_tiles = sorted(set(associated_map.keys()) - set(valid_tiles))
	for key in invalid_tiles:
		associated_map[key] = None

	#-- read and interpolate DEM to coordinates in parallel
	for t in range(comm.Get_rank(), len(valid_tiles), comm.Get_size()):
		key = valid_tiles[t]
		sub = tile_attrs[key]['tile']
		name = tile_attrs[key]['name']
		#-- read central DEM file (geotiff within gzipped tar file)
		tar = '{0}.tar.gz'.format(name)
		elevation_file = os.path.join(elevation_directory,sub,tar)
		DEM,MASK,FV,xi,yi = read_DEM_file(elevation_file)
		#-- buffer DEM using values from adjacent tiles
		#-- use 200m (10 geosegs and divisible by ArcticDEM and REMA pixels)
		#-- use 750m for GIMP
		bf = 750 if (DEM_MODEL == 'GIMP') else 200
		ny,nx = np.shape(DEM)
		dx = np.abs(xi[1]-xi[0]).astype('i')
		dy = np.abs(yi[1]-yi[0]).astype('i')
		#-- new buffered DEM and mask
		d = np.full((ny+2*bf//dy,nx+2*bf//dx),FV,dtype=np.float32)
		m = np.ones((ny+2*bf//dy,nx+2*bf//dx),dtype=np.bool)
		d[bf//dy:-bf//dy,bf//dx:-bf//dx] = DEM.copy()
		m[bf//dy:-bf//dy,bf//dx:-bf//dx] = MASK.copy()
		DEM,MASK = (None,None)
		#-- new buffered image x and y coordinates
		x = (xi[0] - bf) + np.arange((nx+2*bf//dx))*dx
		y = (yi[0] - bf) + np.arange((ny+2*bf//dy))*dy
		#-- min and max of left column, center column, right column
		XL,XC,XR = [[xi[0]-bf,xi[0]-dx],[xi[0],xi[-1]],[xi[-1]+dx,xi[-1]+bf]]
		xlimits = [XL,XL,XL,XC,XC,XR,XR,XR] #-- LLLCCRRR
		#-- min and max of bottom row, middle row, top row
		YB,YM,YT = [[yi[0]-bf,yi[0]-dy],[yi[0],yi[-1]],[yi[-1]+dy,yi[-1]+bf]]
		ylimits = [YB,YM,YT,YB,YT,YB,YM,YT] #-- BMTBTBMT

		#-- buffer using neighbor tiles (REMA/GIMP) or sub-tiles (ArcticDEM)
		if (DEM_MODEL == 'REMA'):
			#-- REMA tiles to read to buffer the image
			IMy,IMx = np.array(re.findall('(\d+)_(\d+)',sub).pop(),dtype='i')
			#-- neighboring tiles for buffering DEM (LB,LM,LT,CB,CT,RB,RM,RT)
			xtiles = [IMx-1,IMx-1,IMx-1,IMx,IMx,IMx+1,IMx+1,IMx+1] #-- LLLCCRRR
			ytiles = [IMy-1,IMy,IMy+1,IMy-1,IMy+1,IMy-1,IMy,IMy+1] #-- BMTBTBMT
			for xtl,ytl,xlim,ylim in zip(xtiles,ytiles,xlimits,ylimits):
				#-- read DEM file (geotiff within gzipped tar file)
				bkey = '{0:02d}_{1:02d}'.format(ytl,xtl)
				#-- if buffer file is a valid tile within the DEM
				#-- if file doesn't exist: will be all fill value with all mask
				if bkey in tile_attrs.keys():
					bsub = tile_attrs[bkey]['tile']
					btar = '{0}.tar.gz'.format(tile_attrs[bkey]['name'])
					buffer_file = os.path.join(elevation_directory,bkey,btar)
					if os.access(buffer_file, os.F_OK):
						DEM,MASK,FV,x1,y1=read_DEM_buffer(buffer_file,xlim,ylim)
						xmin = np.int((x1[0] - x[0])//dx)
						xmax = np.int((x1[-1] - x[0])//dx) + 1
						ymin = np.int((y1[0] - y[0])//dy)
						ymax = np.int((y1[-1] - y[0])//dy) + 1
						#-- add to buffered DEM and mask
						d[ymin:ymax,xmin:xmax] = DEM.copy()
						m[ymin:ymax,xmin:xmax] = MASK.copy()
						DEM,MASK = (None,None)
		elif (DEM_MODEL == 'GIMP'):
			#-- GIMP tiles to read to buffer the image
			IMx,IMy = np.array(re.findall('(\d+)_(\d+)',sub).pop(),dtype='i')
			#-- neighboring tiles for buffering DEM (LB,LM,LT,CB,CT,RB,RM,RT)
			xtiles = [IMx-1,IMx-1,IMx-1,IMx,IMx,IMx+1,IMx+1,IMx+1] #-- LLLCCRRR
			ytiles = [IMy-1,IMy,IMy+1,IMy-1,IMy+1,IMy-1,IMy,IMy+1] #-- BMTBTBMT
			for xtl,ytl,xlim,ylim in zip(xtiles,ytiles,xlimits,ylimits):
				#-- read DEM file (geotiff within gzipped tar file)
				bkey = '{0:d}_{1:d}'.format(xtl,ytl)
				#-- if buffer file is a valid tile within the DEM
				#-- if file doesn't exist: will be all fill value with all mask
				if bkey in tile_attrs.keys():
					bsub = tile_attrs[bkey]['tile']
					btar = '{0}.tar.gz'.format(tile_attrs[bkey]['name'])
					buffer_file = os.path.join(elevation_directory,bkey,btar)
					if os.access(buffer_file, os.F_OK):
						DEM,MASK,FV,x1,y1=read_DEM_buffer(buffer_file,xlim,ylim)
						xmin = np.int((x1[0] - x[0])//dx)
						xmax = np.int((x1[-1] - x[0])//dx) + 1
						ymin = np.int((y1[0] - y[0])//dy)
						ymax = np.int((y1[-1] - y[0])//dy) + 1
						#-- add to buffered DEM and mask
						d[ymin:ymax,xmin:xmax] = DEM.copy()
						m[ymin:ymax,xmin:xmax] = MASK.copy()
						DEM,MASK = (None,None)
		elif (DEM_MODEL == 'ArcticDEM'):
			#-- ArcticDEM sub-tiles to read to buffer the image
			#-- extract parameters from tile filename
			IMy,IMx,STx,STy,res,vers = rx1.findall(name).pop()
			IMy,IMx,STx,STy = np.array([IMy,IMx,STx,STy],dtype='i')
			#-- neighboring tiles for buffering DEM (LB,LM,LT,CB,CT,RB,RM,RT)
			#-- LLLCCRRR
			xtiles = [IMx+(STx-2)//2,IMx+(STx-2)//2,IMx+(STx-2)//2,IMx,IMx,
				IMx+STx//2,IMx+STx//2,IMx+STx//2]
			xsubtiles = [(STx-2) % 2 + 1,(STx-2) % 2 + 1,(STx-2) % 2 + 1,
				STx,STx,STx % 2 + 1,STx % 2 + 1,STx % 2 + 1]
			#-- BMTBTBMT
			ytiles = [IMy+(STy-2)//2,IMy,IMy+STy//2,IMy+(STy-2)//2,
				IMy+STy//2,IMy+(STy-2)//2,IMy,IMy+STy//2]
			ysubtiles = [(STy-2) % 2 + 1,STy,STy % 2 + 1,(STy-2) % 2 + 1,
				STy % 2 + 1,(STy-2) % 2 + 1,STy,STy % 2 + 1]
			#-- for each buffer tile and sub-tile
			kwargs = (xtiles,ytiles,xsubs,ysubs,xlimits,ylimits)
			for xtl,ytl,xs,ys,xlim,ylim in zip(*kwargs):
				#-- read DEM file (geotiff within gzipped tar file)
				args = (ytl,xtl,xs,ys,res,vers)
				bkey = '{0:02d}_{1:02d}_{2}_{3}'.format(*args)
				#-- if buffer file is a valid sub-tile within the DEM
				#-- if file doesn't exist: all fill value with all mask
				if bkey in tile_attrs.keys():
					bsub = tile_attrs[bkey]['tile']
					btar = '{0}.tar.gz'.format(tile_attrs[bkey]['name'])
					buffer_file = os.path.join(elevation_directory,bsub,btar)
					if os.access(buffer_file, os.F_OK):
						DEM,MASK,FV,x1,y1=read_DEM_buffer(buffer_file,xlim,ylim)
						xmin = np.int((x1[0] - x[0])//dx)
						xmax = np.int((x1[-1] - x[0])//dx) + 1
						ymin = np.int((y1[0] - y[0])//dy)
						ymax = np.int((y1[-1] - y[0])//dy) + 1
						#-- add to buffered DEM and mask
						d[ymin:ymax,xmin:xmax] = DEM.copy()
						m[ymin:ymax,xmin:xmax] = MASK.copy()
						DEM,MASK = (None,None)

		#-- indices of x and y coordinates within tile
		tile_indices, = np.nonzero(associated_map[key])
		#-- use spline interpolation to calculate DEM values at coordinates
		f1 = scipy.interpolate.RectBivariateSpline(x,y,d.T,kx=1,ky=1)
		f2 = scipy.interpolate.RectBivariateSpline(x,y,m.T,kx=1,ky=1)
		dataout = f1.ev(X[tile_indices],Y[tile_indices])
		maskout = f2.ev(X[tile_indices],Y[tile_indices])
		#-- save DEM to output variables
		distributed_dem.data[tile_indices] = dataout
		distributed_dem.mask[tile_indices] = maskout.astype(np.bool)
		#-- clear DEM and mask variables
		f1,f2,dataout,maskout,d,m = (None,None,None,None,None,None)

	#-- communicate output MPI matrices between ranks
	#-- operations are element summations and logical "and" across elements
	comm.Allreduce(sendbuf=[distributed_dem.data, MPI.FLOAT], \
		recvbuf=[dem_h.data, MPI.FLOAT], op=MPI.SUM)
	comm.Allreduce(sendbuf=[distributed_dem.mask, MPI.BOOL], \
		recvbuf=[dem_h.mask, MPI.BOOL], op=MPI.LAND)
	distributed_dem = None
	#-- wait for all distributed processes to finish for beam
	comm.Barrier()

	#-- output to file
	if (comm.rank == 0) and bool(valid_tiles):
		#-- output interpolated DEM to HDF5
		dem_h.data[dem_h.mask] = dem_h.fill_value
		#-- output text file
		fileBasename,fileExtension = os.path.splitext(FILE)
		output_file = '{0}_{1}{2}'.format(fileBasename,DEM_MODEL,fileExtension)
		#-- print file information
		print('\t{0}'.format(output_file)) if VERBOSE else None
		#-- print interpolated data to file
		fid = open(output_file,'w')
		for i in range(n_pts):
			args = (latitude[i],longitude[i],dem_h[i])
			print('{0:0.6f},{1:0.6f},{2:0.6f}'.format(*args),file=fid)
		#-- close the output file
		fid.close()

#-- run main program
if __name__ == '__main__':
	main()
