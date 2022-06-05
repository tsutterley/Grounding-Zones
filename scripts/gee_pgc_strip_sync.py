#!/usr/bin/env python
u"""
gee_pgc_strip_sync.py
Written by Tyler Sutterley (06/2022)

Syncs Reference Elevation Map of Antarctica (REMA) DEM or ArcticDEM
    strip tar files from Google Earth Engine

CALLING SEQUENCE:
    python gee_pgc_strip_sync.py --model REMA --year 2014 --scale 32

COMMAND LINE OPTIONS:
    --help: list the command line options
    -m X, --model: PGC digital elevation model
        ArcticDEM
        REMA
    -v X, --version X: DEM version
        v1.0 (default)
        v3.0
    -r X, --resolution X: DEM spatial resolution
        8m
        2m (default)
    -S X, --scale X: Output spatial resolution (resampled)
        default is the same as the original strip
    -Y X, --year X: Year of DEM strips to sync (default=All)
    -B X, --bbox X: Bounding box for spatial query
    -R X, --restart X: Indice for restarting PGC DEM sync
    -A X, --active X: Number of currently active tasks allowed
    -M, --matchtag: Output matchtag raster files
    -I, --index: Output index shapefiles

PYTHON DEPENDENCIES:
    ee: Python bindings for calling the Earth Engine API
        https://github.com/google/earthengine-api

UPDATE HISTORY:
    Updated 06/2022: added restart and bbox command line options
    Written 05/2022
"""
from __future__ import print_function

import ee
import time
import logging
import argparse

# PURPOSE: sync local PGC DEM strip files with Google Earth Engine
def gee_pgc_strip_sync(model, version, resolution,
    YEARS=None,
    BOUNDS=None,
    START=0,
    ACTIVE=1,
    SCALE=None,
    MATCHTAG=False,
    INDEX=False):

    # initialize Google Earth Engine API
    ee.Initialize()
    # standard logging output
    logging.basicConfig(level=logging.INFO)
    # format version and scale
    VERSION = version[:2].upper()
    if not SCALE:
        SCALE = int(resolution[:-1])
    # image collection with PGC DEM strip data
    collection = ee.ImageCollection(f'UMN/PGC/{model}/{VERSION}/{resolution}')
    # for each year of strip data
    for _, year in enumerate(YEARS):
        # reduce image collection to year
        filtered = collection.filterDate(f'{year}-01-01', f'{year}-12-31')
        # reduce image collection to spatial bounding box
        if BOUNDS is not None:
            filtered = filtered.filterBounds(ee.Geometry.BBox(*BOUNDS))
        # number of images in filtered image collection
        n_images = filtered.size().getInfo()
        # create list from filtered image collection
        collection_list = filtered.toList(n_images)
        # for each image in the list
        # (can restart at a particular image)
        for i in range(START,n_images):
            img = ee.Image(collection_list.get(i))
            granule = img.id().getInfo()
            # log granule
            logging.info(granule)
            properties = img.getInfo()['properties']
            # get elevation geotiff
            elev = img.select('elevation')
            # calculate DEM standard deviation
            maxPixels = (SCALE//int(resolution[:-1]))**2
            stdev = elev.reduceResolution(ee.Reducer.sampleStdDev(),
                maxPixels=maxPixels)
            # combine elevation and standard deviation
            image = ee.Image.cat([elev, stdev.float()])
            # scale and reduce resolution
            # export to google drive
            task = ee.batch.Export.image.toDrive(**{
                'image': image,
                'description': f'{granule}_{SCALE}m_{version}_dem',
                'scale': SCALE,
                'fileFormat': 'GeoTIFF',
                'folder': f'{model}_{SCALE}m',
                'formatOptions': {'cloudOptimized': True}
            })
            task.start()
            # limit number of currently active tasks
            task_limiter = (((i-START) % ACTIVE) == 0)
            while task.active() and task_limiter:
                time.sleep(1)

            # output the DEM matchtag raster file
            if MATCHTAG:
                # get matchtag geotiff
                matchtag = img.select('matchtag')
                # export to google drive
                task = ee.batch.Export.image.toDrive(**{
                    'image': matchtag,
                    'description': f'{granule}_{SCALE}m_{version}_matchtag',
                    'scale': SCALE,
                    'fileFormat': 'GeoTIFF',
                    'folder': f'{model}_{SCALE}m',
                    'formatOptions': {'cloudOptimized': True}
                })
                task.start()

            # output the DEM index shapefiles
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
                    'folder': f'{model}_{SCALE}m',
                })
                task.start()

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Syncs Reference Elevation Map of Antarctica
            (REMA) DEM or ArcticDEM strip tar files from Google
            Earth Engine
            """
    )
    # command line parameters
    # DEM model
    parser.add_argument('--model', '-m',
        type=str, choices=('ArcticDEM','REMA'), default='REMA',
        help='PGC digital elevation model (DEM)')
    # DEM model version
    parser.add_argument('--version', '-v',
        type=str, choices=('v1.0','v3.0'), default='v1.0',
        help='PGC DEM version')
    # DEM spatial resolution
    parser.add_argument('--resolution', '-r',
        type=str, choices=('2m','8m'), default='2m',
        help='PGC DEM spatial resolution')
    # output spatial resolution
    # default is the same as the original DEM strip
    parser.add_argument('--scale', '-S',
        type=int, help='Output spatial resolution')
    # PGC DEM strip parameters
    # years of PGC DEMs to sync
    parser.add_argument('--year', '-Y',
        type=int, nargs='+', default=range(2014, 2018),
        help='Years of PGC DEM strips to sync')
    # bounding box for reducing image collection
    parser.add_argument('--bbox','-B',
        type=float, nargs=4,
        metavar=('lon_min','lat_min','lon_max','lat_max'),
        help='Bounding box for spatial query')
    # restart sync at indice
    parser.add_argument('--restart', '-R',
        type=int, default=0,
        help='Indice for restarting PGC DEM sync')
    # limit number of currently active tasks
    parser.add_argument('--active', '-A',
        type=int, default=3000,
        help='Number of currently active tasks allowed')
    # output matchtag raster files
    parser.add_argument('--matchtag','-M',
        default=False, action='store_true',
        help='Output PGC DEM matchtag raster files')
    # output index shapefiles
    parser.add_argument('--index','-I',
        default=False, action='store_true',
        help='Output PGC DEM index shapefiles')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()
    # run Google Earth Engine sync
    gee_pgc_strip_sync(args.model, args.version, args.resolution,
        YEARS=args.year,
        BOUNDS=args.bbox,
        START=args.restart,
        ACTIVE=args.active,
        SCALE=args.scale,
        MATCHTAG=args.matchtag,
        INDEX=args.index)

# run main program
if __name__ == '__main__':
    main()
