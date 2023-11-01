#!/usr/bin/env python
u"""
raster.py
Written by Tyler Sutterley (11/2023)

Utilities for reading and operating on raster data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    h5py: Pythonic interface to the HDF5 binary data format
        https://www.h5py.org/
    gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
        https://pypi.python.org/pypi/GDAL
    PyYAML: YAML parser and emitter for Python
        https://github.com/yaml/pyyaml

PROGRAM DEPENDENCIES:
    spatial.py: utilities for reading and writing spatial data

UPDATE HISTORY:
    Updated 11/2023: cache interpolators for improving computational times
    Written 10/2023
"""
import warnings
import collections
import numpy as np
import scipy.interpolate
# attempt imports
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("pyproj not available", ImportWarning)
try:
    import pyTMD.spatial
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.warn("pyTMD not available", ImportWarning)

class raster:
    """Utilities for using raster files
    """
    np.seterr(invalid='ignore')
    def __init__(self, **kwargs):
        # set default class attributes
        self.attributes = dict()
        self.interpolator = collections.OrderedDict()

    # PURPOSE: read a raster file
    def from_file(self, input_file, format=None, **kwargs):
        """
        Read a raster file from an input format

        Parameters
        ----------
        input_file: str
            path or memory map for raster file
        format: str
            format of input file
        **kwargs: dict
            Keyword arguments for file reader
        """
        dinput = pyTMD.spatial.from_file(
            filename=input_file,
            format=format,
            **kwargs
        )
        # separate dimensions from fields
        self.dims, self.fields = ([], [])
        # convert from dictionary to class attributes
        for key,val in dinput.items():
            if key in ('x','y','time'):
                self.dims.append(key)
            elif key in ('attributes','filename'):
                pass
            else:
                self.fields.append(key)
            # set attribute
            setattr(self, key, val)
        # return the raster object
        return self

    def warp(self, xout, yout, order=0, reducer=np.ceil):
        """
        Interpolate raster data to a new grid

        Parameters
        ----------
        datain: np.ndarray
            input data grid to be interpolated
        xin: np.ndarray
            input x-coordinate array (monotonically increasing)
        yin: np.ndarray
            input y-coordinate array (monotonically increasing)
        xout: np.ndarray
            output x-coordinate array
        yout: np.ndarray
            output y-coordinate array
        order: int, default 0
            interpolation order

                - ``0``: nearest-neighbor interpolation
                - ``k``: bivariate spline interpolation of degree k
        reducer: obj, default np.ceil
            operation for converting mask to boolean
        """
        temp = raster()
        # for each field in the input data
        for field in self.fields:
            # extract data for field
            d_in = getattr(self, field)
            # interpolate values
            if (order == 0):
                # interpolate with nearest-neighbors
                xcoord = (len(self.x)-1)*(xout-self.x[0])/(self.x[-1]-self.x[0])
                ycoord = (len(self.y)-1)*(yout-self.y[0])/(self.y[-1]-self.y[0])
                xcoord = np.clip(xcoord, 0, len(self.x)-1)
                ycoord = np.clip(ycoord, 0, len(self.y)-1)
                XI = np.around(xcoord).astype(np.int32)
                YI = np.around(ycoord).astype(np.int32)
                # interpolate data and mask for field
                d_out = d_in[YI, XI]
                if np.ma.is_masked(d_in):
                    mask = reducer(d_in.mask[YI, XI])
                    d_out = np.ma.array(d_out, mask=mask.astype(bool))
                # set interpolated data for field
                setattr(temp, field, d_out)
            else:
                # interpolate with bivariate spline approximations
                # cache interpolator for faster interpolation
                if field not in self.interpolator:
                    self.interpolator[field] = \
                        scipy.interpolate.RectBivariateSpline(
                            self.x, self.y, d_in.T,
                            kx=order, ky=order
                        )
                    if np.ma.is_masked(d_in):
                        self.interpolator[field].mask = \
                            scipy.interpolate.RectBivariateSpline(
                                self.x, self.y, d_in.mask.T,
                                kx=order, ky=order
                            )
                # interpolate data and mask for field
                d_out = self.interpolator[field].ev(xout, yout)
                if np.ma.is_masked(d_in):
                    mask = reducer(self.interpolator[field].mask.ev(xout, yout))
                    d_out = np.ma.array(d_out, mask=mask.astype(bool))
                # set interpolated data for field
                setattr(temp, field, d_out)
        # return the interpolated data on the output grid
        return temp

    def get_latlon(self, srs_proj4=None, srs_wkt=None, srs_epsg=None):
        """
        Get the latitude and longitude of grid cells

        Parameters
        ----------
        srs_proj4: str or NoneType, default None
            PROJ4 projection string
        srs_wkt: str or NoneType, default None
            Well-Known Text (WKT) projection string
        srs_epsg: int or NoneType, default None
            EPSG projection code

        Returns
        -------
        longitude: np.ndarray
            longitude coordinates of grid cells
        latitude: np.ndarray
            latitude coordinates of grid cells
        """
        # set the spatial projection reference information
        if srs_proj4 is not None:
            source = pyproj.CRS.from_proj4(srs_proj4)
        elif srs_wkt is not None:
            source = pyproj.CRS.from_wkt(srs_wkt)
        elif srs_epsg is not None:
            source = pyproj.CRS.from_epsg(srs_epsg)
        else:
            source = pyproj.CRS.from_string(self.projection)
        # target spatial reference (WGS84 latitude and longitude)
        target = pyproj.CRS.from_epsg(4326)
        # create transformation
        transformer = pyproj.Transformer.from_crs(source, target,
            always_xy=True)
        # create meshgrid of points in original projection
        x, y = np.meshgrid(self.x, self.y)
        # convert coordinates to latitude and longitude
        self.lon, self.lat = transformer.transform(x, y)
        return self

    def copy(self):
        """
        Copy a ``raster`` object to a new ``raster`` object
        """
        temp = raster()
        # copy attributes or update attributes dictionary
        if isinstance(self.attributes, list):
            setattr(temp, 'attributes', self.attributes)
        elif isinstance(self.attributes, dict):
            temp.attributes.update(self.attributes)
        # get dimensions and field names
        temp.dims = self.dims
        temp.fields = self.fields
        # assign variables to self
        for key in [*self.dims, *self.fields]:
            try:
                setattr(temp, key, getattr(self, key))
            except AttributeError:
                pass
        return temp

    def flip(self, axis=0):
        """
        Reverse the order of data and dimensions along an axis

        Parameters
        ----------
        axis: int, default 0
            axis to reorder
        """
        # output spatial object
        temp = self.copy()
        # copy dimensions and reverse order
        if (axis == 0):
            temp.y = temp.y[::-1].copy()
        elif (axis == 1):
            temp.x = temp.x[::-1].copy()
        # attempt to reverse possible data variables
        for key in self.fields:
            try:
                setattr(temp, key, np.flip(getattr(self, key), axis=axis))
            except Exception as exc:
                pass
        return temp
