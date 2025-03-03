Grounding-Zones
===============

Python Tools for Estimating Ice Sheet Grounding Zone Locations with data from NASA polar altimetry missions

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    getting_started/Install.rst
    getting_started/Parallel-HDF5.rst
    getting_started/Contributing.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: API Reference

    api_reference/io/ATL14.rst
    api_reference/io/icebridge.rst
    api_reference/io/raster.rst
    api_reference/io/utilities.rst
    api_reference/crs.rst
    api_reference/fit.rst
    api_reference/mosaic.rst
    api_reference/spatial.rst
    api_reference/utilities.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Utilities

    api_reference/filter_ICESat_GLA12.rst
    api_reference/fit_surface_tiles.rst
    api_reference/MPI_median_elevation_filter.rst
    api_reference/MPI_triangulate_elevation.rst
    api_reference/symbolic_icebridge_files.rst
    api_reference/symbolic_ICESat_GLA12.rst
    api_reference/symbolic_ICESat2_files.rst
    api_reference/tile_icebridge_data.rst
    api_reference/tile_ICESat_GLA12.rst
    api_reference/tile_ICESat2_ATL06.rst
    api_reference/tile_ICESat2_ATL11.rst
    api_reference/track_ICESat_GLA12.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Subsetting

    api_reference/subset/MPI_reduce_ICESat2_ATL03_RGI.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL06_RGI.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL11_RGI.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL06_drainages.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL11_drainages.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL06_grounded.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL11_grounded.rst
    api_reference/subset/MPI_reduce_ICESat_GLA12_grounding_zone.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL03_grounding_zone.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL06_grounding_zone.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL11_grounding_zone.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL06_ice_shelves.rst
    api_reference/subset/MPI_reduce_ICESat2_ATL11_ice_shelves.rst
    api_reference/subset/reduce_ICESat_GLA12_raster.rst
    api_reference/subset/reduce_ICESat2_ATL06_raster.rst
    api_reference/subset/reduce_ICESat2_ATL07_raster.rst
    api_reference/subset/reduce_ICESat2_ATL10_raster.rst
    api_reference/subset/reduce_ICESat2_ATL11_raster.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: DAC

    api_reference/DAC/aviso_dac_sync.rst
    api_reference/DAC/calculate_inverse_barometer.rst
    api_reference/DAC/cds_mslp_sync.rst
    api_reference/DAC/interp_DAC_icebridge_data.rst
    api_reference/DAC/interp_DAC_ICESat_GLA12.rst
    api_reference/DAC/interp_IB_response_ICESat_GLA12.rst
    api_reference/DAC/interp_IB_response_ICESat2_ATL06.rst
    api_reference/DAC/interp_IB_response_ICESat2_ATL07.rst
    api_reference/DAC/interp_IB_response_ICESat2_ATL11.rst


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: DEM

    api_reference/DEM/check_DEM_ICESat2_ATL06.rst
    api_reference/DEM/create_GIMP_tile_index.rst
    api_reference/DEM/gee_pgc_strip_sync.rst
    api_reference/DEM/interp_ATL14_DEM_icebridge_data.rst
    api_reference/DEM/interp_ATL14_DEM_ICESat_GLA12.rst
    api_reference/DEM/interp_ATL14_DEM_ICESat2_ATL06.rst
    api_reference/DEM/interp_ATL14_DEM_ICESat2_ATL11.rst
    api_reference/DEM/MPI_interpolate_DEM.rst
    api_reference/DEM/MPI_DEM_ICESat_GLA12.rst
    api_reference/DEM/MPI_DEM_ICESat2_ATL03.rst
    api_reference/DEM/MPI_DEM_ICESat2_ATL06.rst
    api_reference/DEM/MPI_DEM_ICESat2_ATL11.rst
    api_reference/DEM/nsidc_convert_GIMP_DEM.rst
    api_reference/DEM/pgc_arcticdem_strip_sync.rst
    api_reference/DEM/pgc_arcticdem_sync.rst
    api_reference/DEM/pgc_rema_strip_sync.rst
    api_reference/DEM/pgc_rema_sync.rst
    api_reference/DEM/scp_pgc_dem_strips.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Geoid

    api_reference/geoid/compute_geoid_icebridge_data.rst
    api_reference/geoid/compute_geoid_ICESat_GLA12.rst
    api_reference/geoid/compute_geoid_ICESat2_ATL03.rst
    api_reference/geoid/compute_geoid_ICESat2_ATL06.rst
    api_reference/geoid/compute_geoid_ICESat2_ATL07.rst
    api_reference/geoid/compute_geoid_ICESat2_ATL10.rst
    api_reference/geoid/compute_geoid_ICESat2_ATL11.rst
    api_reference/geoid/compute_geoid_ICESat2_ATL12.rst
    api_reference/geoid/interp_EGM2008_icebridge_data.rst
    api_reference/geoid/interp_EGM2008_ICESat_GLA12.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: GZ

    api_reference/GZ/calculate_grounding_zone.rst
    api_reference/GZ/calculate_GZ_ICESat_GLA12.rst
    api_reference/GZ/calculate_GZ_ICESat2_ATL03.rst
    api_reference/GZ/calculate_GZ_ICESat2_ATL06.rst
    api_reference/GZ/calculate_GZ_ICESat2_ATL11.rst
    api_reference/GZ/model_grounding_zone.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: SL

    api_reference/SL/copernicus_sea_level_sync.rst
    api_reference/SL/interp_sea_level_ICESat2_ATL06.rst
    api_reference/SL/interp_sea_level_ICESat2_ATL07.rst
    api_reference/SL/interp_sea_level_ICESat2_ATL11.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Tides

    api_reference/tides/adjust_tides_ICESat2_ATL11.rst
    api_reference/tides/compute_LPET_elevations.rst
    api_reference/tides/compute_LPET_icebridge_data.rst
    api_reference/tides/compute_LPET_ICESat_GLA12.rst
    api_reference/tides/compute_LPET_ICESat2_ATL03.rst
    api_reference/tides/compute_LPET_ICESat2_ATL06.rst
    api_reference/tides/compute_LPET_ICESat2_ATL07.rst
    api_reference/tides/compute_LPET_ICESat2_ATL10.rst
    api_reference/tides/compute_LPET_ICESat2_ATL11.rst
    api_reference/tides/compute_LPET_ICESat2_ATL12.rst
    api_reference/tides/compute_LPT_displacements.rst
    api_reference/tides/compute_LPT_icebridge_data.rst
    api_reference/tides/compute_LPT_ICESat_GLA12.rst
    api_reference/tides/compute_OPT_displacements.rst
    api_reference/tides/compute_OPT_icebridge_data.rst
    api_reference/tides/compute_OPT_ICESat_GLA12.rst
    api_reference/tides/compute_SET_displacements.rst
    api_reference/tides/compute_tidal_currents.rst
    api_reference/tides/compute_tidal_elevations.rst
    api_reference/tides/compute_tides_icebridge_data.rst
    api_reference/tides/compute_tides_ICESat_GLA12.rst
    api_reference/tides/compute_tides_ICESat2_ATL03.rst
    api_reference/tides/compute_tides_ICESat2_ATL06.rst
    api_reference/tides/compute_tides_ICESat2_ATL07.rst
    api_reference/tides/compute_tides_ICESat2_ATL10.rst
    api_reference/tides/compute_tides_ICESat2_ATL11.rst
    api_reference/tides/compute_tides_ICESat2_ATL12.rst
    api_reference/tides/fit_tides_ICESat2_ATL11.rst
    api_reference/tides/interpolate_tide_adjustment.rst
    api_reference/tides/mosaic_tide_adjustment.rst
    api_reference/tides/tidal_constants_ICESat2_ATL11.rst
    api_reference/tides/tidal_histogram_ICESat2_ATL11.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Project Details

    project/Contributors.rst
    project/Licenses.rst
    project/Citations.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Bibliography

    project/Bibliography.rst
