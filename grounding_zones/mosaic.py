#!/usr/bin/env python
u"""
mosaic.py
Written by Tyler Sutterley (08/2023)
Utilities for creating spatial mosaics

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

UPDATE HISTORY:
    Written 08/2023
"""
from __future__ import print_function, annotations

import numpy as np

class mosaic:
    """Utility for creating spatial mosaics
    """
    def __init__(self, **kwargs):
        self.extent = [np.inf,-np.inf,np.inf,-np.inf]
        self.spacing = [None,None]
        self.fill_value = np.nan

    def update_spacing(self, x, y):
        """
        update the step size of mosaic
        """
        try:
            self.spacing = (x[1] - x[0], y[1] - y[0])
        except:
            pass
        return self

    def update_bounds(self, x, y):
        """
        update the bounds of mosaic
        """
        # check that there is data
        if not np.any(x) or not np.any(y):
            return self
        # get extent of new data
        extent = [x.min(), x.max(), y.min(), y.max()]
        if (extent[0] < self.extent[0]):
            self.extent[0] = np.copy(extent[0])
        if (extent[1] > self.extent[1]):
            self.extent[1] = np.copy(extent[1])
        if (extent[2] < self.extent[2]):
            self.extent[2] = np.copy(extent[2])
        if (extent[3] > self.extent[3]):
            self.extent[3] = np.copy(extent[3])
        return self

    def image_coordinates(self, x, y):
        """
        get the image coordinates
        """
        # check that there is data
        if not np.any(x) or not np.any(y):
            return (None, None)
        # get the image coordinates
        iy = np.array((y[:,None] - self.extent[2])/self.spacing[1], dtype=np.int64)
        ix = np.array((x[None,:] - self.extent[0])/self.spacing[0], dtype=np.int64)
        return (iy, ix)

    @property
    def dimensions(self):
        """Dimensions of the mosaic"""
        dims = [None, None]
        # calculate y dimensions with new extents
        dims[0] = np.int64((self.extent[3] - self.extent[2])/self.spacing[1]) + 1
        # calculate x dimensions with new extents
        dims[1] = np.int64((self.extent[1] - self.extent[0])/self.spacing[0]) + 1
        return dims

    @property
    def shape(self):
        """Shape of the mosaic"""
        return (self.dimensions[0], self.dimensions[1], )

    @property
    def x(self):
        """X-coordinates of the mosaic"""
        return self.extent[0] + self.spacing[0]*np.arange(self.dimensions[1])

    @property
    def y(self):
        """Y-coordinates of the mosaic"""
        return self.extent[2] + self.spacing[1]*np.arange(self.dimensions[0])
