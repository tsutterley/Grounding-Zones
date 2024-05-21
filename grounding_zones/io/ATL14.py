#!/usr/bin/env python
u"""
ATL14.py
Written by Tyler Sutterley (05/2024)
Read ICESat-2 Gridded Land Ice Height (ATL14) products

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    mosaic.py: Utilities for creating spatial mosaics
    utilities.py: File read and access utilities

UPDATE HISTORY:
    Written 05/2024: moved ATL14 data inputs to a separate module
"""

from __future__ import print_function, annotations

from io import IOBase
import logging
import pathlib
import numpy as np
from grounding_zones.mosaic import mosaic
from grounding_zones.utilities import import_dependency

# attempt imports
is2tk = import_dependency('icesat2_toolkit')
netCDF4 = import_dependency('netCDF4')
pyproj = import_dependency('pyproj')

def ATL14(DEM_MODEL: str | list | IOBase,
        BOUNDS: list | np.ndarray | None = None,
        BUFFER: int | float = 20
    ):
    """
    Read and mosaic ATL14 DEM model files within spatial bounds

    Parameters
    ----------
    DEM_MODEL: str or list or io.IOBase
        ATL14 DEM model files
    BOUNDS: list or np.ndarray, default 
        spatial bounds to crop DEM model files
    BUFFER: int | float, default 20
        buffer in pixels around the bounds
    """
    # verify ATL14 DEM file is iterable
    if isinstance(DEM_MODEL, (str, IOBase)):
        DEM_MODEL = [DEM_MODEL]

    # subset ATL14 elevation field to bounds
    DEM = mosaic()
    # iterate over each ATL14 DEM file
    for MODEL in DEM_MODEL:
        # check if DEM is an s3 presigned url
        if str(MODEL).startswith('s3:'):
            is2tk.utilities.attempt_login('urs.earthdata.nasa.gov',
                authorization_header=True)
            session = is2tk.utilities.s3_filesystem()
            MODEL = session.open(MODEL, mode='rb')
        elif isinstance(MODEL, IOBase):
            pass
        else:
            MODEL = pathlib.Path(MODEL).expanduser().absolute()

        # open ATL14 DEM file for reading
        logging.info(str(MODEL))
        with netCDF4.Dataset(MODEL, mode='r') as fileID:
            # get original grid coordinates
            x = fileID.variables['x'][:].copy()
            y = fileID.variables['y'][:].copy()
            # fill_value for invalid heights
            fv = fileID['h'].getncattr('_FillValue')
            # get coordinate reference system
            grid_mapping_name = fileID['h'].getncattr('grid_mapping')
            spatial_epsg = int(fileID[grid_mapping_name].spatial_epsg)
        # update the mosaic grid spacing
        DEM.update_spacing(x, y)
        # get size of DEM
        ny, nx = len(y), len(x)
        # set coordinate reference system
        setattr(DEM, 'crs', pyproj.CRS.from_epsg(spatial_epsg))

        # get maximum bounds of DEM
        if BOUNDS is None:
            # use complete bounds of dataset
            indx, indy = slice(None, None, 1), slice(None, None, 1)
        else:
            # determine buffered bounds of data in image coordinates
            # (affine transform)
            IMxmin = int((BOUNDS[0] - x[0])//DEM.spacing[0]) - BUFFER
            IMxmax = int((BOUNDS[1] - x[0])//DEM.spacing[0]) + BUFFER
            IMymin = int((BOUNDS[2] - y[0])//DEM.spacing[1]) - BUFFER
            IMymax = int((BOUNDS[3] - y[0])//DEM.spacing[1]) + BUFFER
            # get buffered bounds of data
            # and convert invalid values to 0
            indx = slice(np.maximum(IMxmin,0), np.minimum(IMxmax,nx), 1)
            indy = slice(np.maximum(IMymin,0), np.minimum(IMymax,ny), 1)
        # update bounds using input coordinates
        DEM.update_bounds(x[indx], y[indy])
    
    # check that DEM has a valid shape
    if np.any(np.sign(DEM.shape) == -1):
        raise ValueError('Values outside of ATL14 range')

    # fill ATL14 to mosaic
    DEM.h = np.ma.zeros(DEM.shape, dtype=np.float32, fill_value=fv)
    DEM.h_sigma2 = np.ma.zeros(DEM.shape, dtype=np.float32, fill_value=fv)
    DEM.ice_area = np.ma.zeros(DEM.shape, dtype=np.float32, fill_value=fv)
    # iterate over each ATL14 DEM file
    for MODEL in DEM_MODEL:
        # check if DEM is an s3 presigned url
        if str(MODEL).startswith('s3:'):
            is2tk.utilities.attempt_login('urs.earthdata.nasa.gov',
                authorization_header=True)
            session = is2tk.utilities.s3_filesystem()
            MODEL = session.open(MODEL, mode='rb')
        else:
            MODEL = pathlib.Path(MODEL).expanduser().absolute()

        # open ATL14 DEM file for reading
        fileID = netCDF4.Dataset(MODEL, mode='r')
        # get original grid coordinates
        x = fileID.variables['x'][:].copy()
        y = fileID.variables['y'][:].copy()
        # get size of DEM
        ny, nx = len(y), len(x)

        # get bounds of dataset
        if BOUNDS is None:
            # use complete bounds of dataset
            indx, indy = slice(None, None, 1), slice(None, None, 1)
        else:
            # determine buffered bounds of data in image coordinates
            # (affine transform)
            IMxmin = int((BOUNDS[0] - x[0])//DEM.spacing[0]) - BUFFER
            IMxmax = int((BOUNDS[1] - x[0])//DEM.spacing[0]) + BUFFER
            IMymin = int((BOUNDS[2] - y[0])//DEM.spacing[1]) - BUFFER
            IMymax = int((BOUNDS[3] - y[0])//DEM.spacing[1]) + BUFFER
            # get buffered bounds of data
            # and convert invalid values to 0
            indx = slice(np.maximum(IMxmin,0), np.minimum(IMxmax,nx), 1)
            indy = slice(np.maximum(IMymin,0), np.minimum(IMymax,ny), 1)
        # get the image coordinates of the input file
        iy, ix = DEM.image_coordinates(x[indx], y[indy])

        # create mosaic of DEM variables
        if np.any(iy) and np.any(ix):
            DEM.h[iy, ix] = fileID['h'][indy, indx]
            DEM.h_sigma2[iy, ix] = fileID['h_sigma'][indy, indx]**2
            DEM.ice_area[iy, ix] = fileID['ice_area'][indy, indx]
        # close the ATL14 file
        fileID.close()

    # update masks for DEM
    for key in ['h', 'h_sigma2', 'ice_area']:
        val = getattr(DEM, key)
        val.mask = (val.data == val.fill_value) | np.isnan(val.data)
        val.data[val.mask] = val.fill_value

    # return the DEM object
    return DEM
