#!/usr/bin/env python
u"""
gee_rema_strip_sync.py
Written by Tyler Sutterley (05/2022)

Syncs Reference Elevation Map of Antarctica (REMA) DEM strip tar files
    from Google Earth Engine

CALLING SEQUENCE:
    python gee_rema_strip_sync.py --version v1.0 --resolution 8m

COMMAND LINE OPTIONS:
    --help: list the command line options
    -v X, --version X: REMA DEM version
        v1.0 (default)
    -r X, --resolution X: REMA DEM spatial resolution
        8m
        2m (default)
    -s X, --scale X: Output spatial resolution (resampled)
        default is the same as the original REMA strip
    -t X, --tolerance X: Tolerance for differences between fields
    -Y X, --year X: Year of REMA DEM strips to sync (default=All)
    -M, --matchtag: Output REMA matchtag raster files
    -I, --index: Output REMA index shapefiles

PYTHON DEPENDENCIES:
    ee: Python bindings for calling the Earth Engine API
        https://github.com/google/earthengine-api

UPDATE HISTORY:
    Written 05/2022
"""
from __future__ import print_function

import ee
import logging
import argparse

# PURPOSE: sync local REMA strip files with Google Earth Engine
def gee_rema_strip_sync(version, resolution, YEARS=None,
    SCALE=None, TOLERANCE=None, MATCHTAG=False, INDEX=False):
    # initialize Google Earth Engine API
    ee.Initialize()
    #-- standard logging output
    logging.basicConfig(level=logging.INFO)
    # format version and scale
    VERSION = version[:2].upper()
    if not SCALE:
        SCALE = int(resolution[:-1])
    # image collection with REMA strip data
    collection = ee.ImageCollection(f'UMN/PGC/REMA/{VERSION}/{resolution}')
    # for each year of strip data
    for _, year in enumerate(YEARS):
        # reduce image collection to year
        filtered = collection.filterDate(f'{year}-01-01', f'{year}-12-31')
        n_images = filtered.size().getInfo()
        # convert image collection to list
        collection_list = filtered.toList(n_images)
        # for each image in the list
        for i in range(n_images):
            img = ee.Image(collection_list.get(i))
            granule = img.id().getInfo()
            # log granule
            logging.info(granule)
            properties = img.getInfo()['properties']
            # get elevation geotiff
            elev = img.select('elevation')
            # scale and reduce resolution
            # convolve with a gaussian kernel
            # find where maximum deviations are less than tolerance
            if SCALE != int(resolution[:-1]):
                w_smooth = SCALE/int(resolution[:-1])/2.0
                kernel = ee.Kernel.gaussian(radius=w_smooth, normalize=True)
                smooth = elev.convolve(kernel)
                absdiff = elev.subtract(smooth).abs()
                elev = elev.updateMask(absdiff.lt(TOLERANCE))
            # export to google drive
            task = ee.batch.Export.image.toDrive(**{
                'image': elev,
                'description': f'{granule}_{SCALE}m_{version}_dem',
                'scale': SCALE,
                'fileFormat': 'GeoTIFF',
                'folder': f'REMA_{SCALE}m',
                'formatOptions': {'cloudOptimized': True}
            })
            task.start()

            # output the REMA matchtag raster file
            if MATCHTAG:
                # get matchtag geotiff
                matchtag = img.select('matchtag')
                # scale and reduce resolution
                if SCALE != int(resolution[:-1]):
                    matchtag = matchtag.updateMask(absdiff.lt(TOLERANCE))
                # export to google drive
                task = ee.batch.Export.image.toDrive(**{
                    'image': matchtag,
                    'description': f'{granule}_{SCALE}m_{version}_matchtag',
                    'scale': SCALE,
                    'fileFormat': 'GeoTIFF',
                    'folder': f'REMA_{SCALE}m',
                    'formatOptions': {'cloudOptimized': True}
                })
                task.start()

            # output the REMA index shapefiles
            if INDEX:
                # get coordinates of footprint linear ring
                coordinates = ee.Geometry.LinearRing(
                    properties['system:footprint']['coordinates'])
                features = ee.FeatureCollection([coordinates])
                # export to google drive
                task = ee.batch.Export.table.toDrive(**{
                    'collection': features,
                    'description': f'{granule}_{SCALE}m_{version}_index',
                    'fileFormat': 'SHP',
                    'folder': f'REMA_{SCALE}m',
                })
                task.start()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Syncs Reference Elevation Map of Antarctica (REMA)
            DEM strip tar files from Google Earth Engine
            """
    )
    # command line parameters
    # REMA DEM model version
    parser.add_argument('--version', '-v',
        type=str, choices=('v1.0',), default='v1.0',
        help='REMA DEM version')
    # DEM spatial resolution
    parser.add_argument('--resolution', '-r',
        type=str, choices=('2m','8m'), default='2m',
        help='REMA DEM spatial resolution')
    # output spatial resolution
    # default is the same as the original REMA strip
    parser.add_argument('--scale', '-s',
        type=int, help='Output spatial resolution')
    # tolerance for differences between smoothed fields and original
    parser.add_argument('--tolerance', '-t',
        type=float, default=5.0,
        help='Tolerance for differences between fields')
    # REMA strip parameters
    parser.add_argument('--year', '-Y',
        type=int, nargs='+', default=range(2014, 2018),
        help='Years of REMA DEM strips to sync')
    # output matchtag raster files
    parser.add_argument('--matchtag','-M',
        default=False, action='store_true',
        help='Output REMA matchtag raster files')
    # output index shapefiles
    parser.add_argument('--index','-I',
        default=False, action='store_true',
        help='Output REMA index shapefiles')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()
    # run Google Earth Engine sync
    gee_rema_strip_sync(args.version, args.resolution,
        YEARS=args.year, SCALE=args.scale, TOLERANCE=args.tolerance,
        MATCHTAG=args.matchtag, INDEX=args.index)

# run main program
if __name__ == '__main__':
    main()
