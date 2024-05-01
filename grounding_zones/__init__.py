"""
A Grounding Zone toolkit for Python
===================================

grounding_zones contains Python tools for estimating ice sheet grounding zone
locations with the NASA Ice, Cloud and land Elevation Satellite-2 (ICESat-2)

The package works using Python packages (numpy, scipy, scikit-learn, shapely)
combined with data storage in HDF5 and zarr, and mapping with
matplotlib and cartopy

Documentation is available at https://grounding-zones.readthedocs.io
"""
import grounding_zones.crs
import grounding_zones.fit
from grounding_zones import io
from grounding_zones.mosaic import mosaic
import grounding_zones.utilities
import grounding_zones.version

# get version number
__version__ = grounding_zones.version.version
