#!/usr/bin/env python
u"""
nsidc_convert_GIMP_DEM.py
Written by Tyler Sutterley (05/2022)

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
    -D X, --directory X: working data directory for output GIMP files
    -U X, --user X: username for NASA Earthdata Login
    -W X, --password X: password for NASA Earthdata Login
    -N X, --netrc X: path to .netrc file for alternative authentication
    -v X, --version X: data release of the GIMP dataset
    -M X, --mode X: Local permissions mode of the directories and files synced

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    lxml: Pythonic XML and HTML processing library using libxml2/libxslt
        https://lxml.de/
        https://github.com/lxml/lxml
    future: Compatibility layer between Python 2 and Python 3
        (http://python-future.org/)
    gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
        https://pypi.python.org/pypi/GDAL/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 07/2022: place GDAL import within try/except statement
    Updated 05/2022: use argparse descriptions within documentation
    Updated 04/2021: set a default netrc file and check access
        default credentials from environmental variables
    Updated 01/2021: using utilities modules to list and download from server
        using argparse to set command line options
    Updated 09/2019: output shapefiles for each tile similar to REMA/ArcticDEM
        added ssl context to urlopen headers
    Written 09/2019
"""
from __future__ import print_function

import os
import re
import io
import uuid
import netrc
import tarfile
import getpass
import logging
import builtins
import argparse
import warnings
import posixpath
import lxml.etree
import grounding_zones.utilities
# attempt imports
try:
    import osgeo.gdal, osgeo.osr, osgeo.ogr
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("GDAL not available")
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read GIMP image mosaic and output as gzipped tar file
def nsidc_convert_GIMP_DEM(base_dir, VERSION, MODE=0o775):

    # remote https server
    remote_dir = posixpath.join('https://n5eil01u.ecs.nsidc.org',
        'MEASURES',f'NSIDC-0645.{VERSION:03.0f}','2003.02.20')
    # regex pattern to find files and extract tile coordinates
    rx = re.compile(rf'gimpdem(\d+_\d+)_v{VERSION:04.1f}.tif$')
    # image resolution
    res = '30m'

    # compile HTML parser for lxml
    parser = lxml.etree.HTMLParser()

    # read and parse request for remote files (columns and dates)
    colnames,collastmod,_ = grounding_zones.utilities.nsidc_list(
        remote_dir, build=False, parser=parser, pattern=rx,sort=True)

    # read each GIMP DEM file
    for colname,remote_mtime in zip(colnames,collastmod):
        # print input file to track progress
        logging.info(colname)
        # extract tile number
        tile, = rx.findall(colname)
        # recursively create directories if not currently available
        d = os.path.join(base_dir,'GIMP','30m',tile)
        os.makedirs(d,MODE) if not os.access(d,os.F_OK) else None

        # Create and submit request. There are a wide range of exceptions
        # that can be thrown here, including HTTPError and URLError.
        # chunked transfer encoding size
        CHUNK = 16 * 1024
        # copy contents to BytesIO object using chunked transfer encoding
        # transfer should work properly with ascii and binary data formats
        fileID,_ = grounding_zones.utilities.from_nsidc(
            posixpath.join(remote_dir, colname),
            build=False, chunk=CHUNK)
        # rewind retrieved binary to start of file
        fileID.seek(0)

        # open gzipped tar file
        FILE = colname.replace('.tif','.tar.gz')
        tar = tarfile.open(name=os.path.join(d,FILE),mode='w:gz')
        # add directory
        subdir = f'{tile}_{res}'
        info1 = tarfile.TarInfo(name=subdir)
        info1.type = tarfile.DIRTYPE
        info1.mtime = remote_mtime
        info1.mode = MODE
        tar.addfile(tarinfo=info1)
        # add in memory file to tar
        output = f'{tile}_{res}_dem.tif'
        info2 = tarfile.TarInfo(name=posixpath.join(subdir,output))
        info2.size = fileID.getbuffer().nbytes
        info2.mtime = remote_mtime
        info2.mode = MODE
        tar.addfile(tarinfo=info2, fileobj=fileID)
        # add index subdirectory
        info3 = tarfile.TarInfo(name=posixpath.join(subdir,'index'))
        info3.type = tarfile.DIRTYPE
        info3.mtime = remote_mtime
        info3.mode = MODE
        tar.addfile(tarinfo=info3)

        # rewind retrieved binary to start of file
        fileID.seek(0)
        # use GDAL memory-mapped file to read dem
        mmap_name = f"/vsimem/{uuid.uuid4().hex}"
        osgeo.gdal.FileFromMemBuffer(mmap_name, fileID.read())
        dataset = osgeo.gdal.Open(mmap_name)

        # get dimensions of tile
        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize
        # dx and dy
        info_geotiff = dataset.GetGeoTransform()
        # no data value
        fill_value = dataset.GetRasterBand(1).GetNoDataValue()
        fill_value = 0.0 if (fill_value is None) else fill_value
        # get coordinates of tile
        xmin = info_geotiff[0]
        ymax = info_geotiff[3]
        xmax = xmin + info_geotiff[1]*(xsize-1)
        ymin = ymax + info_geotiff[5]*(ysize-1)
        # create linear ring x and y coordinates
        xbox = [xmin,xmax,xmax,xmin,xmin]
        ybox = [ymin,ymin,ymax,ymax,ymin]
        # st_area_sh: tile area (meters^2)
        st_area_sh = (xmax-xmin)*(ymax-ymin)
        # st_length_: perimeter length of tile (meters)
        st_length_ = int(2*(xmax-xmin) + 2*(ymax-ymin))
        # close the dataset
        dataset = None
        osgeo.gdal.Unlink(mmap_name)

        # save DEM tile outlines to ESRI shapefile
        driver = osgeo.ogr.GetDriverByName('Esri Shapefile')
        # use GDAL memory-mapped file
        mmap = {}
        for key in ('dbf','prj','shp','shx'):
            mmap[key] = f"/vsimem/{tile}_{res}_index.{key}"
        # create memory-mapped shapefile
        ds = driver.CreateDataSource(mmap['shp'])
        # set the spatial reference info
        # EPSG: 3413 (NSIDC Sea Ice Polar Stereographic North, WGS84)
        SpatialReference = osgeo.osr.SpatialReference()
        SpatialReference.ImportFromEPSG(3413)
        layer = ds.CreateLayer('', SpatialReference, osgeo.ogr.wkbPolygon)
        # Add shapefile attributes (following attributes from ArcticDEM and REMA)
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
        # Create a new feature (attribute and geometry)
        feature = osgeo.ogr.Feature(defn)
        # Add shapefile attributes
        feature.SetField('DEM_ID', f'{tile}_{res}')
        feature.SetField('DEM_NAME', colname)
        feature.SetField('TILE', tile)
        feature.SetField('ND_VALUE', fill_value)
        feature.SetField('DEM_RES', info_geotiff[1])
        feature.SetField('REL_VER', VERSION)
        feature.SetField('REG_SRC', 'ICESat')
        feature.SetField('ST_AREA', st_area_sh)
        feature.SetField('ST_LENGTH', st_length_)
        # create LineString object and add x/y points
        ring_obj = osgeo.ogr.Geometry(osgeo.ogr.wkbLinearRing)
        for x,y in zip(xbox,ybox):
            ring_obj.AddPoint(x,y)
        # create Polygon object for LineString of tile
        poly_obj = osgeo.ogr.Geometry(osgeo.ogr.wkbPolygon)
        poly_obj.AddGeometry(ring_obj)
        feature.SetGeometry(poly_obj)
        layer.CreateFeature(feature)
        # set geometry and features to None
        feature.Destroy()
        poly_obj = None
        # Save and close everything
        ds.Destroy()
        layer = None
        feature = None
        fileID = None

        # add in memory shapefile to tar
        for key in ('dbf','prj','shp','shx'):
            output = f'{tile}_{res}_index.{key}'
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

        # close tar file
        tar.close()
        # set permissions level to MODE
        os.chmod(os.path.join(d,FILE), MODE)
        # keep remote modification time of directory and local access time
        os.utime(d, (os.stat(d).st_atime, remote_mtime))

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Reads GIMP 30m DEM tiles from the OSU Greenland
            Ice Mapping Project (GIMP) and outputs as gzipped tar files
            """
    )
    # command line parameters
    # NASA Earthdata credentials
    parser.add_argument('--user','-U',
        type=str, default=os.environ.get('EARTHDATA_USERNAME'),
        help='Username for NASA Earthdata Login')
    parser.add_argument('--password','-W',
        type=str, default=os.environ.get('EARTHDATA_PASSWORD'),
        help='Password for NASA Earthdata Login')
    parser.add_argument('--netrc','-N',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.path.join(os.path.expanduser('~'),'.netrc'),
        help='Path to .netrc file for authentication')
    # working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # GIMP data version
    parser.add_argument('--version','-v',
        type=float, default=1.1,
        help='GIMP Data Version')
    # print information about processing run
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of processing run')
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[args.verbose])

    # NASA Earthdata hostname
    HOST = 'urs.earthdata.nasa.gov'
    # get authentication
    if not args.user and not os.access(args.netrc,os.F_OK):
        # check that NASA Earthdata credentials were entered
        args.user = builtins.input(f'Username for {HOST}: ')
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')
    elif os.access(args.netrc, os.F_OK):
        args.user,_,args.password = netrc.netrc(args.netrc).authenticators(HOST)
    elif args.user and not args.password:
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')

    # build a urllib opener for NSIDC
    # Add the username and password for NASA Earthdata Login system
    grounding_zones.utilities.build_opener(args.user,args.password)

    # check internet connection before attempting to run program
    # check NASA earthdata credentials before attempting to run program
    if grounding_zones.utilities.check_credentials():
        nsidc_convert_GIMP_DEM(args.directory, args.version, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
