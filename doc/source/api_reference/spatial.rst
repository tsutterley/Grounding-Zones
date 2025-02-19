=======
spatial
=======

Coordinates Reference System (CRS) and spatial transformation routines

 - Can read netCDF4, HDF5, (cloud optimized) geotiff or (geo)parquet files

`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/grounding_zones/spatial.py

General Methods
===============


.. autofunction:: grounding_zones.spatial.case_insensitive_filename

.. autofunction:: grounding_zones.spatial.from_file

.. autofunction:: grounding_zones.spatial.from_ascii

.. autofunction:: grounding_zones.spatial.from_netCDF4

.. autofunction:: grounding_zones.spatial.from_HDF5

.. autofunction:: grounding_zones.spatial.from_geotiff

.. autofunction:: grounding_zones.spatial.from_parquet

.. autofunction:: grounding_zones.spatial.to_file

.. autofunction:: grounding_zones.spatial.to_ascii

.. autofunction:: grounding_zones.spatial.to_netCDF4

.. autofunction:: grounding_zones.spatial._drift_netCDF4

.. autofunction:: grounding_zones.spatial._grid_netCDF4

.. autofunction:: grounding_zones.spatial._time_series_netCDF4

.. autofunction:: grounding_zones.spatial.to_HDF5

.. autofunction:: grounding_zones.spatial.to_geotiff

.. autofunction:: grounding_zones.spatial.to_parquet

.. autofunction:: grounding_zones.spatial.expand_dims

.. autofunction:: grounding_zones.spatial.default_field_mapping

.. autofunction:: grounding_zones.spatial.inverse_mapping
