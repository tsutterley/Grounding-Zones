#!/usr/bin/env python
u"""
nsidc_convert_GIMP_DEM.py
Written by Tyler Sutterley (09/2019)

Reads GIMP 30m DEM tiles from the OSU Greenland Ice Mapping Project
	https://nsidc.org/data/nsidc-0645/versions/1
Outputs as gzipped tar files similar to REMA and ArcticDEM tiles

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
	-D X, --directory=X: working data directory for output GIMP files
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
	Updated 09/2019: output shapefiles for each tile similar to REMA/ArcticDEM
		added ssl context to urlopen headers
	Written 09/2019
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
import tarfile
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
		HOST = 'https://n5eil01u.ecs.nsidc.org/'
		urllib2.urlopen(HOST,timeout=20,context=ssl.SSLContext())
	except urllib2.URLError:
		raise RuntimeError('Check internet connection')
	else:
		return True

#-- PURPOSE: read GIMP image mosaic and output as gzipped tar file
def nsidc_convert_GIMP_DEM(base_dir,VERSION,USER='',PASSWORD='',MODE=0o775):

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

	#-- find GIMP DEM files
	remote_dir = posixpath.join(HOST,*REMOTE)
	request = urllib2.Request(url=remote_dir)
	#-- read and parse request for remote files (columns and dates)
	tree = lxml.etree.parse(urllib2.urlopen(request), parser)
	colnames = tree.xpath('//td[@class="indexcolname"]//a/@href')
	collastmod = tree.xpath('//td[@class="indexcollastmod"]/text()')
	remote_file_lines = [i for i,f in enumerate(colnames) if rx.match(f)]

	#-- read each GIMP DEM file
	for i in remote_file_lines:
		#-- print input file to track progress
		print(colnames[i])
		#-- extract tile number
		tile, = rx.findall(colnames[i])
		#-- recursively create directories if not currently available
		d = os.path.join(base_dir,'GIMP','30m',tile)
		os.makedirs(d,MODE) if not os.access(d,os.F_OK) else None
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

		#-- open gzipped tar file
		FILE = colnames[i].replace('.tif','.tar.gz')
		tar = tarfile.open(name=os.path.join(d,FILE),mode='w:gz')
		#-- add directory
		subdir = '{0}_{1}'.format(tile,'30m')
		info1 = tarfile.TarInfo(name=subdir)
		info1.type = tarfile.DIRTYPE
		info1.mtime = remote_mtime
		info1.mode = MODE
		tar.addfile(tarinfo=info1)
		#-- add in memory file to tar
		output = '{0}_{1}_dem.tif'.format(tile,'30m')
		info2 = tarfile.TarInfo(name=posixpath.join(subdir,output))
		info2.size = fileID.getbuffer().nbytes
		info2.mtime = remote_mtime
		info2.mode = MODE
		tar.addfile(tarinfo=info2, fileobj=fileID)
		#-- add index subdirectory
		info3 = tarfile.TarInfo(name=posixpath.join(subdir,'index'))
		info3.type = tarfile.DIRTYPE
		info3.mtime = remote_mtime
		info3.mode = MODE
		tar.addfile(tarinfo=info3)

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

		#-- save DEM tile outlines to ESRI shapefile
		driver = osgeo.ogr.GetDriverByName('Esri Shapefile')
		#-- use GDAL memory-mapped file
		mmap = {}
		for key in ('dbf','prj','shp','shx'):
			mmap[key] = "/vsimem/{0}_{1}_index.{2}".format(tile,'30m',key)
		#-- create memory-mapped shapefile
		ds = driver.CreateDataSource(mmap['shp'])
		#-- set the spatial reference info
		#-- EPSG: 3413 (NSIDC Sea Ice Polar Stereographic North, WGS84)
		SpatialReference = osgeo.osr.SpatialReference()
		SpatialReference.ImportFromEPSG(3413)
		layer = ds.CreateLayer('', SpatialReference, osgeo.ogr.wkbPolygon)
		#-- Add shapefile attributes (following attributes from ArcticDEM and REMA)
		layer.CreateField(osgeo.ogr.FieldDefn('DEM_ID', osgeo.ogr.OFTString))
		layer.CreateField(osgeo.ogr.FieldDefn('DEM_NAME', osgeo.ogr.OFTString))
		layer.CreateField(osgeo.ogr.FieldDefn('TILE', osgeo.ogr.OFTString))
		layer.CreateField(osgeo.ogr.FieldDefn('ND_VALUE', osgeo.ogr.OFTReal))
		layer.CreateField(osgeo.ogr.FieldDefn('DEM_RES', osgeo.ogr.OFTInteger))
		layer.CreateField(osgeo.ogr.FieldDefn('REL_VER', osgeo.ogr.OFTReal))
		layer.CreateField(osgeo.ogr.FieldDefn('REG_SRC', osgeo.ogr.OFTString))
		field_area = osgeo.ogr.FieldDefn('ST_AREA', osgeo.ogr.OFTReal)
		field_area.SetWidth(24)
		field_area.SetPrecision(10)
		layer.CreateField(field_area)
		layer.CreateField(osgeo.ogr.FieldDefn('ST_LENGTH', osgeo.ogr.OFTInteger))
		defn = layer.GetLayerDefn()
		#-- Create a new feature (attribute and geometry)
		feature = osgeo.ogr.Feature(defn)
		#-- Add shapefile attributes
		feature.SetField('DEM_ID', '{0}_{1}'.format(tile,'30m'))
		feature.SetField('DEM_NAME', colnames[i])
		feature.SetField('TILE', tile)
		feature.SetField('ND_VALUE', fill_value)
		feature.SetField('DEM_RES', info_geotiff[1])
		feature.SetField('REL_VER', VERSION)
		feature.SetField('REG_SRC', 'ICESat')
		feature.SetField('ST_AREA', st_area_sh)
		feature.SetField('ST_LENGTH', st_length_)
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
		#-- Save and close everything
		ds.Destroy()
		layer = None
		feature = None
		fileID = None

		#-- add in memory shapefile to tar
		for key in ('dbf','prj','shp','shx'):
			output = '{0}_{1}_index.{2}'.format(tile,'30m',key)
			info4 = tarfile.TarInfo(name=posixpath.join(subdir,'index',output))
			info4.size = osgeo.gdal.VSIStatL(mmap[key]).size
			info4.mtime = remote_mtime
			info4.mode = MODE
			fp = osgeo.gdal.VSIFOpenL(mmap[key],'rb')
			with io.BytesIO() as fileID:
				fileID.write(osgeo.gdal.VSIFReadL(1, info4.size, fp))
				fileID.seek(0)
				tar.addfile(tarinfo=info4,fileobj=fileID)
			osgeo.gdal.VSIFCloseL(fp)
			osgeo.gdal.Unlink(mmap[key])

		#-- close tar file
		tar.close()
		#-- set permissions level to MODE
		os.chmod(os.path.join(d,FILE), MODE)
		#-- keep remote modification time of directory and local access time
		os.utime(d, (os.stat(d).st_atime, remote_mtime))

#-- PURPOSE: help module to describe the optional input parameters
def usage():
	print('\nHelp: {0}'.format(os.path.basename(sys.argv[0])))
	print(' -D X, --directory=X\tWorking data directory')
	print(' -R X, --release=X\tData release of the GIMP dataset')
	print(' -U X, --user=X\t\tUsername for NASA Earthdata Login')
	print(' -M X, --mode=X\t\tPermission mode of output file\n')

#-- Main program that calls nsidc_convert_GIMP_DEM()
def main():
	#-- Read the system arguments listed after the program
	long_options=['help','user=','release=','directory=','mode=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'hU:R:D:M:',long_options)

	#-- command line parameters
	USER = ''
	#-- data release
	VERSION = 1.1
	#-- working data directory
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
		nsidc_convert_GIMP_DEM(DIRECTORY, VERSION, USER=USER,
			PASSWORD=PASSWORD, MODE=MODE)

#-- run main program
if __name__ == '__main__':
	main()
