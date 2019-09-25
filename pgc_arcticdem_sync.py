#!/usr/bin/env python
u"""
pgc_arcticdem_sync.py
Written by Tyler Sutterley (05/2019)

Program to sync ArcticDEM tar files
	http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic

CALLING SEQUENCE:
	python pgc_arcticdem_sync.py --version=v3.0 --resolution=2m

COMMAND LINE OPTIONS:
	--help: list the command line options
	-D X, --directory=X: Working data directory (default=$PYTHONPATH)
	-V X, --version=X: ArcticDEM DEM version
		v2.0
		v3.0 (default)
	-R X, --resolution=X: ArcticDEM spatial resolution
		1km
		500m
		100m
		32m
		10m
		2m (default)
	-T X, --tile=X: ArcticDEM tiles to sync (default=All)
	-L, --list: print files to be transferred, but do not execute transfer
	-l, --log: output log of files downloaded
	--clobber: Overwrite existing data in transfer
	-M X, --mode=X: Local permissions mode of the directories and files synced

PYTHON DEPENDENCIES:
	lxml: Pythonic XML and HTML processing library using libxml2/libxslt
		http://lxml.de/
		https://github.com/lxml/lxml
	future: Compatibility layer between Python 2 and Python 3
		(http://python-future.org/)

UPDATE HISTORY:
	Updated 05/2019: sort tile directories of interest.  copy tile shapefiles
	Updated 04/2019: check python version for urllib compatibility with python 2
	Written 04/2019
"""
from __future__ import print_function

import sys
import os
import re
import getopt
import shutil
import posixpath
import lxml.etree
import calendar, time
if sys.version_info[0] == 2:
	import urllib2
else:
	import urllib.request as urllib2

#-- PURPOSE: check internet connection
def check_connection():
	#-- attempt to connect to public http Polar Geospatial Center host
	try:
		urllib2.urlopen('http://data.pgc.umn.edu/elev/dem/', timeout=1)
	except urllib2.URLError:
		raise RuntimeError('Check internet connection')
	else:
		return True

#-- PURPOSE: sync local ArcticDEM files with PGC public server
def pgc_arcticdem_sync(base_dir, VERSION, RESOLUTION, TILES=None,
	LOG=False, LIST=False, CLOBBER=False, MODE=None):
	#-- data directory
	DIRECTORY = os.path.join(base_dir,'ArcticDEM')
	#-- check if directory exists and recursively create if not
	os.makedirs(DIRECTORY,MODE) if not os.path.exists(DIRECTORY) else None

	#-- create log file with list of synchronized files (or print to terminal)
	if LOG:
		#-- format: PGC_ArcticDEM_sync_2002-04-01.log
		today = time.strftime('%Y-%m-%d',time.localtime())
		LOGFILE = 'PGC_ArcticDEM_sync_{0}.log'.format(today)
		fid = open(os.path.join(DIRECTORY,LOGFILE),'w')
		print('PGC ArcticDEM Sync Log ({0})'.format(today), file=fid)
		print('VERSION={0}'.format(VERSION), file=fid)
		print('RESOLUTION={0}'.format(RESOLUTION), file=fid)
		print('TILES={0}'.format(','.join(TILES)), file=fid) if TILES else None
	else:
		#-- standard output (terminal output)
		fid = sys.stdout

	#-- remote http server for PGC DEM data
	HOST = posixpath.join('http://data.pgc.umn.edu','elev','dem','setsm')
	#-- compile regular expression operators for tiles
	R1 = re.compile('|'.join(TILES)) if TILES else re.compile('\d+_\d+')
	R2 = re.compile('(\d+_\d+)_(.*?)\.tar\.gz')
	#-- compile regular expression operators for shapefiles
	R3 = re.compile('(.*?)_Tile_Index_Rel(\d+)\.zip')

	#-- compile HTML parser for lxml
	parser = lxml.etree.HTMLParser()

	#-- remote directory for data version and resolution
	remote_dir = posixpath.join(HOST,'ArcticDEM','mosaic',VERSION,RESOLUTION)
	#-- open connection with PGC server at remote directory
	request = urllib2.Request(remote_dir)
	response = urllib2.urlopen(request, timeout=20)
	#-- read and parse request for files (names and modified dates)
	tree = lxml.etree.parse(response, parser)
	colnames = tree.xpath('//td[@class="indexcolname"]//a/@href')
	collastmod = tree.xpath('//td[@class="indexcollastmod"]/text()')
	remote_sub = sorted([d for i,d in enumerate(colnames) if R1.match(d)])
	#-- for each tile subdirectory
	for sd in remote_sub:
		#-- check if data directory exists and recursively create if not
		local_dir = os.path.join(DIRECTORY,sd)
		if not os.access(local_dir, os.F_OK) and not LIST:
			os.makedirs(local_dir,MODE)
		#-- open connection with PGC server at remote directory
		request = urllib2.Request(posixpath.join(remote_dir,sd))
		response = urllib2.urlopen(request, timeout=20)
		#-- read and parse request for files (names and modified dates)
		tree = lxml.etree.parse(response, parser)
		colnames = tree.xpath('//td[@class="indexcolname"]//a/@href')
		collastmod = tree.xpath('//td[@class="indexcollastmod"]/text()')
		remote_file_lines = [i for i,f in enumerate(colnames) if R2.match(f)]
		#-- sync each ArcticDEM data file
		for i in remote_file_lines:
			#-- remote and local versions of the file
			remote_file = posixpath.join(remote_dir,sd,colnames[i])
			local_file = os.path.join(local_dir,colnames[i])
			#-- get last modified date and convert into unix time
			lastmodtime = time.strptime(collastmod[i].rstrip(),'%Y-%m-%d %H:%M')
			remote_mtime = calendar.timegm(lastmodtime)
			#-- sync ArcticDEM tar file
			http_pull_file(fid, remote_file, remote_mtime, local_file, LIST,
				CLOBBER, MODE)
		#-- close request
		request = None

	#-- remote directory for shapefiles of data version
	remote_dir = posixpath.join(HOST,'ArcticDEM','indexes')
	#-- open connection with PGC server at remote directory
	request = urllib2.Request(remote_dir)
	response = urllib2.urlopen(request, timeout=20)
	#-- read and parse request for files (names and modified dates)
	tree = lxml.etree.parse(response, parser)
	colnames = tree.xpath('//td[@class="indexcolname"]//a/@href')
	collastmod = tree.xpath('//td[@class="indexcollastmod"]/text()')
	remote_file_lines = [i for i,d in enumerate(colnames) if R3.match(d)]
	#-- sync each ArcticDEM shapefile
	for i in remote_file_lines:
		#-- remote and local versions of the file
		remote_file = posixpath.join(remote_dir,colnames[i])
		local_file = os.path.join(DIRECTORY,colnames[i])
		#-- get last modified date and convert into unix time
		lastmodtime = time.strptime(collastmod[i].rstrip(),'%Y-%m-%d %H:%M')
		remote_mtime = calendar.timegm(lastmodtime)
		#-- sync ArcticDEM shapefile
		http_pull_file(fid, remote_file, remote_mtime, local_file, LIST,
			CLOBBER, MODE)

	#-- close log file and set permissions level to MODE
	if LOG:
		fid.close()
		os.chmod(os.path.join(DIRECTORY,LOGFILE), MODE)

#-- PURPOSE: pull file from a remote host checking if file exists locally
#-- and if the remote file is newer than the local file
def http_pull_file(fid,remote_file,remote_mtime,local_file,LIST,CLOBBER,MODE):
	#-- if file exists in file system: check if remote file is newer
	TEST = False
	OVERWRITE = ' (clobber)'
	#-- check if local version of file exists
	if os.access(local_file, os.F_OK):
		#-- check last modification time of local file
		local_mtime = os.stat(local_file).st_mtime
		#-- if remote file is newer: overwrite the local file
		if (remote_mtime > local_mtime):
			TEST = True
			OVERWRITE = ' (overwrite)'
	else:
		TEST = True
		OVERWRITE = ' (new)'
	#-- if file does not exist locally, is to be overwritten, or CLOBBER is set
	if TEST or CLOBBER:
		#-- Printing files transferred
		print('{0} --> '.format(remote_file), file=fid)
		print('\t{0}{1}\n'.format(local_file,OVERWRITE), file=fid)
		#-- if executing copy command (not only printing the files)
		if not LIST:
			#-- Create and submit request. There are a wide range of exceptions
			#-- that can be thrown here, including HTTPError and URLError.
			request = urllib2.Request(remote_file)
			response = urllib2.urlopen(request, timeout=20)
			#-- chunked transfer encoding size
			CHUNK = 16 * 1024
			#-- copy contents to local file using chunked transfer encoding
			#-- transfer should work properly with ascii and binary data formats
			with open(local_file, 'wb') as f:
				shutil.copyfileobj(response, f, CHUNK)
			#-- keep remote modification time of file and local access time
			os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
			os.chmod(local_file, MODE)

#-- PURPOSE: help module to describe the optional input parameters
def usage():
	print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
	print(' -D X, --directory=X\tWorking Data Directory')
	print(' -V X, --version=X\tArcticDEM version')
	print('\tv2.0\n\tv3.0')
	print(' -R X, --resolution=X\tArcticDEM spatial resolution')
	print('\t1km\n\t500m\n\t100m\n\t32m\n\t10m\n\t2m')
	print(' -T X, --tile=X\t\tArcticDEM tiles to sync')
	print(' -L, --list\t\tOnly print files that are to be transferred')
	print(' --clobber\t\tOverwrite existing data in transfer')
	print(' -M X, --mode=X\t\tPermission mode of directories and files synced')
	print(' -l, --log\t\tOutput log file')
	today = time.strftime('%Y-%m-%d',time.localtime())
	LOGFILE = 'PGC_ArcticDEM_sync_{0}.log'.format(today)
	print('    Log file format: {}\n'.format(LOGFILE))

#-- Main program that calls pgc_arcticdem_sync()
def main():
	#-- Read the system arguments listed after the program
	long_options = ['help','directory=','version=','resolution=','tile=',
		'list','log','clobber','mode=']
	optlist,arglist = getopt.getopt(sys.argv[1:],'hD:V:R:T:lLCM:',long_options)

	#-- command line parameters
	base_dir = os.getcwd()
	LIST = False
	LOG = False
	CLOBBER = False
	#-- ArcticDEM parameters
	VERSION = 'v3.0'
	RESOLUTION = '2m'
	TILES = None
	#-- permissions mode of the local directories and files (number in octal)
	MODE = 0o775
	for opt, arg in optlist:
		if opt in ('-h','--help'):
			usage()
			sys.exit()
		elif opt in ("-D","--directory"):
			base_dir = os.path.expanduser(arg)
		elif opt in ("-L","--list"):
			LIST = True
		elif opt in ("-l","--log"):
			LOG = True
		elif opt in ("-C","--clobber"):
			CLOBBER = True
		elif opt in ("-V","--version"):
			VERSION = arg
		elif opt in ("-R","--resolution"):
			RESOLUTION = arg
		elif opt in ("-T","--tile"):
			TILES = arg.split(',')
		elif opt in ("-M","--mode"):
			MODE = int(arg, 8)

	#-- check internet connection before attempting to run program
	if check_connection():
		pgc_arcticdem_sync(base_dir, VERSION, RESOLUTION, TILES=TILES,
			LIST=LIST, LOG=LOG, CLOBBER=CLOBBER, MODE=MODE)

#-- run main program
if __name__ == '__main__':
	main()
