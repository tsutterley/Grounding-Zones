#!/usr/bin/env python
u"""
utilities.py
Written by Tyler Sutterley (05/2024)
File read and access utilities

PYTHON DEPENDENCIES:
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/

UPDATE HISTORY:
    Written 05/2024
"""
from __future__ import annotations

import time
import pathlib
from grounding_zones.utilities import import_dependency

# attempt imports
h5py = import_dependency('h5py')
netCDF4 = import_dependency('netCDF4')

# PURPOSE: attempt to open an HDF5 file and wait if already open
def multiprocess_h5py(filename, *args, **kwargs):
    """
    Open an HDF5 file with a hold for already open files

    Parameters
    ----------
    filename: str
        HDF5 file to open
    args: tuple
        additional arguments to pass to ``h5py.File``
    kwargs: dict
        additional keyword arguments to pass to ``h5py.File``
    """
    # set default keyword arguments
    kwargs.setdefault('mode', 'r')
    # check that file exists if entering with read mode
    filename = pathlib.Path(filename).expanduser().absolute()
    if kwargs['mode'] in ('r','r+') and not filename.exists():
        raise FileNotFoundError(filename)
    # attempt to open HDF5 file
    while True:
        try:
            fileID = h5py.File(filename, *args, **kwargs)
            break
        except (IOError, BlockingIOError, PermissionError) as exc:
            time.sleep(1)
    # return the file access object
    return fileID

# PURPOSE: attempt to open a netCDF4 file and wait if already open
def multiprocess_netCDF4(filename, *args, **kwargs):
    """
    Open a netCDF4 file with a hold for already open files

    Parameters
    ----------
    filename: str
        netCDF4 file to open
    args: tuple
        additional arguments to pass to ``netCDF4.Dataset``
    kwargs: dict
        additional keyword arguments to pass to ``netCDF4.Dataset``
    """
    # set default keyword arguments
    kwargs.setdefault('mode', 'r')
    # check that file exists if entering with read mode
    filename = pathlib.Path(filename).expanduser().absolute()
    if kwargs['mode'] in ('r','r+') and not filename.exists():
        raise FileNotFoundError(filename)
    # attempt to open netCDF4 file
    while True:
        try:
            fileID = netCDF4.Dataset(filename, *args, **kwargs)
            break
        except (IOError, OSError, PermissionError) as e:
            time.sleep(1)
    # return the file access object
    return fileID
