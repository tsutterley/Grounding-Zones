#!/usr/bin/env python
u"""
create_GIMP_tile_index.py
Written by Tyler Sutterley (05/2024)

Reads GIMP 30m DEM tiles from the OSU Greenland Ice Mapping Project
    https://nsidc.org/data/nsidc-0645/versions/1
Creates a single shapefile with the extents of each tile

Reads tiles directly server:
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
    Updated 05/2024: use wrapper to importlib for optional dependencies
    Updated 07/2023: using pathlib to define and operate on paths
    Updated 12/2022: single implicit import of grounding zone tools
    Updated 07/2022: place GDAL import within try/except statement
    Updated 05/2022: use argparse descriptions within documentation
    Updated 04/2021: set a default netrc file and check access
        default credentials from environmental variables
    Updated 01/2021: using utilities modules to list and download from server
        using argparse to set command line options
    Updated 09/2019: remove suffix from name attribute.  add perimeter and area
        added ssl context to urlopen headers
    Written 04/2019
"""
from __future__ import print_function

import os
import re
import time
import uuid
import zipfile
import logging
import pathlib
import argparse
import posixpath
import lxml.etree
import grounding_zones as gz

# attempt imports
osgeo = gz.utilities.import_dependency('osgeo')
osgeo.gdal = gz.utilities.import_dependency('osgeo.gdal')
osgeo.osr = gz.utilities.import_dependency('osgeo.osr')
osgeo.ogr = gz.utilities.import_dependency('osgeo.ogr')

# PURPOSE: read GIMP image mosaic and create tile shapefile
def create_GIMP_tile_index(base_dir, VERSION, MODE=0o775):
    # Directory Setup
    base_dir = pathlib.Path(base_dir).expanduser().absolute()
    ddir = base_dir.joinpath('GIMP', '30m')
    # recursively create directories if not currently available
    ddir.mkdir(mode=MODE, parents=True, exist_ok=True)
    # remote https server
    remote_dir = posixpath.join('https://n5eil01u.ecs.nsidc.org',
        'MEASURES',f'NSIDC-0645.{VERSION:03.0f}','2003.02.20')
    # regex pattern to find files and extract tile coordinates
    rx = re.compile(rf'gimpdem(\d+_\d+)_v{VERSION:04.1f}.tif$')
    # output file format
    ff = 'gimpdem_Tile_Index_Rel{0:0.1f}.{1}'

    # compile HTML parser for lxml
    parser = lxml.etree.HTMLParser()

     # save DEM tile outlines to ESRI shapefile
    driver = osgeo.ogr.GetDriverByName('Esri Shapefile')
    output_shapefile = ddir.joinpath(ff.format(VERSION,'shp'))
    ds = driver.CreateDataSource(str(output_shapefile))
    # set the spatial reference info
    # EPSG: 3413 (NSIDC Sea Ice Polar Stereographic North, WGS84)
    SpatialReference = osgeo.osr.SpatialReference()
    SpatialReference.ImportFromEPSG(3413)
    layer = ds.CreateLayer('', SpatialReference, osgeo.ogr.wkbPolygon)
    # Add shapefile attributes (following attributes from ArcticDEM and REMA)
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

    # create a counter for shapefile id
    id = 1
    # read and parse request for remote files (columns and dates)
    colnames,collastmod,_ = gz.utilities.nsidc_list(remote_dir,
        build=False, parser=parser, pattern=rx, sort=True)

    # read each GIMP DEM file
    for colname, remote_mtime in zip(colnames, collastmod):
        # print input file to track progress
        logging.info(colname)
        # extract tile number
        tile, = rx.findall(colname)

        # Create and submit request. There are a wide range of exceptions
        # that can be thrown here, including HTTPError and URLError.
        # chunked transfer encoding size
        CHUNK = 16 * 1024
        # copy contents to BytesIO object using chunked transfer encoding
        # transfer should work properly with ascii and binary data formats
        fileID,_ = gz.utilities.from_nsidc(
            posixpath.join(remote_dir,colname),
            build=False, chunk=CHUNK)
        # rewind retrieved binary to start of file
        fileID.seek(0)

        # use GDAL memory-mapped file to read dem
        mmap_name = f'/vsimem/{uuid.uuid4().hex}'
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

        # Create a new feature (attribute and geometry)
        feature = osgeo.ogr.Feature(defn)
        # Add shapefile attributes
        feature.SetField('id', id)
        feature.SetField('name', colname.replace('.tif',''))
        feature.SetField('tile', tile)
        feature.SetField('nd_value', fill_value)
        feature.SetField('resolution', info_geotiff[1])
        lastmodtm = time.strftime('%Y-%m-%d', time.gmtime(remote_mtime))
        feature.SetField('lastmodtm', lastmodtm)
        feature.SetField('fileurl', posixpath.join(remote_dir,colname))
        feature.SetField('rel_ver', VERSION)
        feature.SetField('spec_type', 'DEM')
        feature.SetField('reg_src', 'ICESat')
        feature.SetField('st_area_sh', st_area_sh)
        feature.SetField('st_length_', st_length_)

        # create LineString object and add x/y points
        ring_obj = osgeo.ogr.Geometry(osgeo.ogr.wkbLinearRing)
        for x, y in zip(xbox, ybox):
            ring_obj.AddPoint(x, y)
        # create Polygon object for LineString of tile
        poly_obj = osgeo.ogr.Geometry(osgeo.ogr.wkbPolygon)
        poly_obj.AddGeometry(ring_obj)
        feature.SetGeometry(poly_obj)
        layer.CreateFeature(feature)
        # set geometry and features to None
        feature.Destroy()
        poly_obj = None
        # add to counter
        id += 1

    # Save and close everything
    ds.Destroy()
    layer = None
    feature = None
    poly_obj = None

    # create zip file of shapefile objects
    output_zipfile = ddir.joinpath(ff.format(VERSION,'zip'))
    zp = zipfile.ZipFile(str(output_zipfile), mode='w')
    for s in ['dbf','prj','shp','shx']:
        # change the permissions mode of the output shapefiles
        output_file = ddir.joinpath(ff.format(VERSION,s))
        output_file.chmod(mode=MODE)
        # write to zip file
        zp.write(output_file, output_file.name)
    # change the permissions mode of the output zip file
    output_zipfile.chmod(mode=MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Reads GIMP 30m DEM tiles from the OSU Greenland
            Ice Mapping Project (GIMP) and creates a single shapefile
            with the extents of each tile
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
        type=pathlib.Path, default=pathlib.Path.home().joinpath('.netrc'),
        help='Path to .netrc file for authentication')
    # working data directory
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
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
    # build a urllib opener for NASA Earthdata
    # check internet connection before attempting to run program
    opener = gz.utilities.attempt_login(HOST,
        username=args.user, password=args.password,
        netrc=args.netrc)

    # check internet connection before attempting to run program
    # check NASA earthdata credentials before attempting to run program
    if gz.utilities.check_credentials():
        create_GIMP_tile_index(args.directory, args.version, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
