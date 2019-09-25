#!/usr/bin/env python
u"""
make_GIMP_tile_shapefile.py
Written by Tyler Sutterley (09/2019)

Reads GIMP 30m DEM tiles from the OSU Greenland Ice Mapping Project
	https://nsidc.org/data/nsidc-0645/versions/1
Creates a single shapefile with the extents of each tile similar to ArcticDEM

Reads tiles directly from NSIDC server:
https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
https://nsidc.org/support/faq/what-options-are-available-bulk-downloading-data-
	https-earthdata-login-enabled
http://www.voidspace.org.uk/python/articles/authentication.shtml#base64

Register with NASA Earthdata Login system:
https://urs.earthdata.nasa.gov

Add NSIDC_DATAPOOL_OPS to NASA Earthdata Applications
https://urs.earthdata.nasa.gov/oauth/authorize?client_id=_JLuwMHxb2xX6NwYTb4dRA

COMMAND LINE OPTIONS:
	-D X, --directory=X: working data directory for output GIMP shapefile
	-R X, --release=X: data release of the GIMP dataset
	-U X, --user=X: username for NASA Earthdata Login
	-M X, --mode=X: Local permissions mode of the directories and files synced

PYTHON DEPENDENCIES:
	lxml: Pythonic XML and HTML processing library using libxml2/libxslt
		http://lxml.de/
		https://github.com/lxml/lxml
	future: Compatibility layer between Python 2 and Python 3
		(http://python-future.org/)
	gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
		https://pypi.python.org/pypi/GDAL/

UPDATE HISTORY:
	Updated 09/2019: remove suffix from name attribute.  add perimeter and area
		added ssl context to urlopen headers
	Written 04/2019
"""
from __future__ import print_function

import sys
import os
import re
import io
import uuid
import getopt
import shutil
import base64
import zipfile
import getpass
import builtins
import posixpath
import lxml.etree
import calendar, time
import osgeo.gdal, osgeo.osr, osgeo.ogr
if sys.version_info[0] == 2:
	from cookielib import CookieJar
	import urllib2
else:
	from http.cookiejar import CookieJar
	import urllib.request as urllib2

#-- PURPOSE: check internet connection
def check_connection():
	#-- attempt to connect to https host for NSIDC
	try:
		HOST = 'https://n5eil01u.ecs.nsidc.org'
		urllib2.urlopen(HOST,timeout=20,context=ssl.SSLContext())
	except urllib2.URLError:
		raise RuntimeError('Check internet connection')
	else:
		return True

#-- PURPOSE: read GIMP image mosaic and create tile shapefile
def make_GIMP_tile_shapefile(base_dir,VERSION,USER='',PASSWORD='',MODE=0o775):
	#-- Directory Setup
	ddir = os.path.join(base_dir,'GIMP','30m')
	#-- recursively create directories if not currently available
	os.makedirs(ddir) if not os.access(ddir,os.F_OK) else None

	#-- https://docs.python.org/3/howto/urllib2.html#id5
	#-- create a password manager
	password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
	#-- Add the username and password for NASA Earthdata Login system
	password_mgr.add_password(None, 'https://urs.earthdata.nasa.gov',
		USER, PASSWORD)
	#-- Encode username/password for request authorization headers
	base64_string = base64.b64encode('{0}:{1}'.format(USER,PASSWORD).encode())
	#-- compile HTML parser for lxml
	parser = lxml.etree.HTMLParser()
	#-- Create cookie jar for storing cookies. This is used to store and return
	#-- the session cookie given to use by the data server (otherwise will just
	#-- keep sending us back to Earthdata Login to authenticate).
	cookie_jar = CookieJar()
	#-- create "opener" (OpenerDirector instance)
	opener = urllib2.build_opener(
		urllib2.HTTPBasicAuthHandler(password_mgr),
		urllib2.HTTPSHandler(context=ssl.SSLContext()),
		urllib2.HTTPCookieProcessor(cookie_jar))
	#-- add Authorization header to opener
	authorization_header = "Basic {0}".format(base64_string.decode())
	opener.addheaders = [("Authorization", authorization_header)]
	#-- Now all calls to urllib2.urlopen use our opener.
	urllib2.install_opener(opener)
	#-- All calls to urllib2.urlopen will now use handler
	#-- Make sure not to include the protocol in with the URL, or
	#-- HTTPPasswordMgrWithDefaultRealm will be confused.

	#-- remote https server for ICESat-2 Data
	HOST = 'https://n5eil01u.ecs.nsidc.org'
	REMOTE = ['MEASURES','NSIDC-0645.{0:03.0f}'.format(VERSION),'2003.02.20']
	#-- regex pattern to find files and extract tile coordinates
	rx = re.compile('gimpdem(\d+_\d+)_v{0:04.1f}.tif$'.format(VERSION))
	#-- output file format
	ff = 'gimpdem_Tile_Index_Rel{0:0.1f}.{1}'

	#-- save DEM tile outlines to ESRI shapefile
	driver = osgeo.ogr.GetDriverByName('Esri Shapefile')
	ds = driver.CreateDataSource(os.path.join(ddir,ff.format(VERSION,'shp')))
	#-- set the spatial reference info
	#-- EPSG: 3413 (NSIDC Sea Ice Polar Stereographic North, WGS84)
	SpatialReference = osgeo.osr.SpatialReference()
	SpatialReference.ImportFromEPSG(3413)
	layer = ds.CreateLayer('', SpatialReference, osgeo.ogr.wkbPolygon)
	#-- Add shapefile attributes (following attributes from ArcticDEM and REMA)
	layer.CreateField(osgeo.ogr.FieldDefn('id', osgeo.ogr.OFTInteger))
	layer.CreateField(osgeo.ogr.FieldDefn('name', osgeo.ogr.OFTString))
	layer.CreateField(osgeo.ogr.FieldDefn('tile', osgeo.ogr.OFTString))
	layer.CreateField(osgeo.ogr.FieldDefn('nd_value', osgeo.ogr.OFTReal))
	layer.CreateField(osgeo.ogr.FieldDefn('resolution', osgeo.ogr.OFTInteger))
	layer.CreateField(osgeo.ogr.FieldDefn('lastmodtm', osgeo.ogr.OFTString))
	layer.CreateField(osgeo.ogr.FieldDefn('fileurl', osgeo.ogr.OFTString))
	layer.CreateField(osgeo.ogr.FieldDefn('rel_ver', osgeo.ogr.OFTReal))
	layer.CreateField(osgeo.ogr.FieldDefn('spec_type', osgeo.ogr.OFTString))
	layer.CreateField(osgeo.ogr.FieldDefn('reg_src', osgeo.ogr.OFTString))
	field_area = osgeo.ogr.FieldDefn('st_area_sh', osgeo.ogr.OFTReal)
	field_area.SetWidth(24)
	field_area.SetPrecision(10)
	layer.CreateField(field_area)
	layer.CreateField(osgeo.ogr.FieldDefn('st_length_', osgeo.ogr.OFTInteger))
	defn = layer.GetLayerDefn()

	#-- find GIMP DEM files
	remote_dir = posixpath.join(HOST,*REMOTE)
	request = urllib2.Request(url=remote_dir)
	#-- read and parse request for remote files (columns and dates)
	tree = lxml.etree.parse(urllib2.urlopen(request), parser)
	colnames = tree.xpath('//td[@class="indexcolname"]//a/@href')
	collastmod = tree.xpath('//td[@class="indexcollastmod"]/text()')
	remote_file_lines = [i for i,f in enumerate(colnames) if rx.match(f)]

	#-- create a counter for shapefile id
	id = 1
	#-- read each GIMP DEM file
	for i in remote_file_lines:
		#-- print input file to track progress
		print(colnames[i])
		#-- extract tile number
		tile, = rx.findall(colnames[i])
		#-- get last modified date and convert into unix time
		lastmodtime = time.strptime(collastmod[i].rstrip(),'%Y-%m-%d %H:%M')
		remote_mtime = calendar.timegm(lastmodtime)

		#-- Create and submit request. There are a wide range of exceptions
		#-- that can be thrown here, including HTTPError and URLError.
		request = urllib2.Request(posixpath.join(remote_dir,colnames[i]))
		#-- open BytesIO object
		fileID = io.BytesIO()
		#-- chunked transfer encoding size
		CHUNK = 16 * 1024
		#-- copy contents to BytesIO object using chunked transfer encoding
		#-- transfer should work properly with ascii and binary data formats
		shutil.copyfileobj(urllib2.urlopen(request), fileID, CHUNK)
		#-- rewind retrieved binary to start of file
		fileID.seek(0)
		#-- use GDAL memory-mapped file to read dem
		mmap_name = "/vsimem/{0}".format(uuid.uuid4().hex)
		osgeo.gdal.FileFromMemBuffer(mmap_name, fileID.read())
		dataset = osgeo.gdal.Open(mmap_name)

		#-- get dimensions of tile
		xsize = dataset.RasterXSize
		ysize = dataset.RasterYSize
		#-- dx and dy
		info_geotiff = dataset.GetGeoTransform()
		#-- no data value
		fill_value = dataset.GetRasterBand(1).GetNoDataValue()
		fill_value = 0.0 if (fill_value is None) else fill_value
		#-- get coordinates of tile
		xmin = info_geotiff[0]
		ymax = info_geotiff[3]
		xmax = xmin + info_geotiff[1]*(xsize-1)
		ymin = ymax + info_geotiff[5]*(ysize-1)
		#-- create linear ring x and y coordinates
		xbox = [xmin,xmax,xmax,xmin,xmin]
		ybox = [ymin,ymin,ymax,ymax,ymin]
		#-- st_area_sh: tile area (meters^2)
		st_area_sh = (xmax-xmin)*(ymax-ymin)
		#-- st_length_: perimeter length of tile (meters)
		st_length_ = int(2*(xmax-xmin) + 2*(ymax-ymin))
		#-- close the dataset
		dataset = None
		osgeo.gdal.Unlink(mmap_name)

		#-- Create a new feature (attribute and geometry)
		feature = osgeo.ogr.Feature(defn)
		#-- Add shapefile attributes
		feature.SetField('id', id)
		feature.SetField('name', colnames[i].replace('.tif',''))
		feature.SetField('tile', tile)
		feature.SetField('nd_value', fill_value)
		feature.SetField('resolution', info_geotiff[1])
		feature.SetField('lastmodtm', time.strftime('%Y-%m-%d',lastmodtime))
		feature.SetField('fileurl', posixpath.join(remote_dir,colnames[i]))
		feature.SetField('rel_ver', VERSION)
		feature.SetField('spec_type', 'DEM')
		feature.SetField('reg_src', 'ICESat')
		feature.SetField('st_area_sh', st_area_sh)
		feature.SetField('st_length_', st_length_)

		#-- create LineString object and add x/y points
		ring_obj = osgeo.ogr.Geometry(osgeo.ogr.wkbLinearRing)
		for x,y in zip(xbox,ybox):
			ring_obj.AddPoint(x,y)
		#-- create Polygon object for LineString of tile
		poly_obj = osgeo.ogr.Geometry(osgeo.ogr.wkbPolygon)
		poly_obj.AddGeometry(ring_obj)
		feature.SetGeometry(poly_obj)
		layer.CreateFeature(feature)
		#-- set geometry and features to None
		feature.Destroy()
		poly_obj = None
		#-- add to counter
		id += 1

	#-- Save and close everything
	ds.Destroy()
	layer = None
	feature = None
	poly_obj = None

	#-- create zip file of shapefile objects
	zp = zipfile.ZipFile(os.path.join(ddir,ff.format(VERSION,'zip')),'w')
	for s in ['dbf','prj','shp','shx']:
		#-- change the permissions mode of the output shapefiles
		os.chmod(os.path.join(ddir,ff.format(VERSION,s)),MODE)
		#-- write to zip file
		zp.write(os.path.join(ddir,ff.format(VERSION,s)),ff.format(VERSION,s))
	#-- change the permissions mode of the output zip file
	os.chmod(os.path.join(ddir,ff.format(VERSION,'zip')),MODE)

#-- PURPOSE: help module to describe the optional input parameters
def usage():
	print('\nHelp: {0}'.format(os.path.basename(sys.argv[0])))
	print(' -D X, --directory=X\tWorking data directory')
	print(' -R X, --release=X\tData release of the GIMP dataset')
	print(' -U X, --user=X\t\tUsername for NASA Earthdata Login')
	print(' -M X, --mode=X\t\tPermission mode of output file\n')

#-- Main program that calls make_GIMP_tile_shapefile()
def main():
	#-- Read the system arguments listed after the program
	long_options=['help','user=','release=','directory=','mode=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'hU:R:D:M:',long_options)

	#-- command line parameters
	USER = ''
	#-- data release
	VERSION = 1.1
	#-- Working data directory
	DIRECTORY = os.getcwd()
	#-- permissions mode of the local directories and files (number in octal)
	MODE = 0o775
	for opt, arg in optlist:
		if opt in ('-h','--help'):
			usage()
			sys.exit()
		elif opt in ("-U","--user"):
			USER = arg
		elif opt in ("-R","--release"):
			VERSION = float(arg)
		elif opt in ("-D","--directory"):
			DIRECTORY = os.path.expanduser(arg)
		elif opt in ("-M","--mode"):
			MODE = int(arg, 8)

	#-- NASA Earthdata hostname
	HOST = 'urs.earthdata.nasa.gov'
	#-- check that NASA Earthdata credentials were entered
	if not USER:
		USER = builtins.input('Username for {0}: '.format(HOST))
	#-- enter password securely from command-line
	PASSWORD = getpass.getpass('Password for {0}@{1}: '.format(USER,HOST))

	#-- check internet connection before attempting to run program
	if check_connection():
		make_GIMP_tile_shapefile(DIRECTORY, VERSION, USER=USER,
			PASSWORD=PASSWORD, MODE=MODE)

#-- run main program
if __name__ == '__main__':
	main()
