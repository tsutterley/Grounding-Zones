#!/usr/bin/env python
u"""
test_tide_corrections.py (01/2024)
Download ATL03 and ATL07 files from NSIDC and compare tides values
"""
import os
import pytest
import warnings
import numpy as np
import grounding_zones as gz

# attempt imports
is2tk = gz.utilities.import_dependency('icesat2_toolkit')
pyTMD = gz.utilities.import_dependency('pyTMD')
timescale = gz.utilities.import_dependency('timescale')

# path to an ATL03 file from NSIDC
ATL03 = ['https://n5eil01u.ecs.nsidc.org','ATLAS','ATL03.006','2018.10.14',
    'ATL03_20181014000347_02350101_006_02.h5']
# PURPOSE: Download ATL03 granule from NSIDC
@pytest.fixture(scope="module", autouse=True)
def download_ATL03(username, password):
    # only download ATL03 file if not currently existing
    if not os.access(ATL03[-1], os.F_OK):
        # download an ATL03 file from NSIDC
        is2tk.utilities.from_nsidc(ATL03, username=username,
            password=password, local=ATL03[-1], verbose=True)
        autoremove = True
    else:
        autoremove = False
    # run tests
    yield
    # remove the local file
    if autoremove:
        os.remove(ATL03[-1])

# PURPOSE: Compare long-period equilibrium tides with ATL03 predicted values
def test_ATL03_equilibrium_tides():
    # path to local ATL03 granule
    granule = ATL03[-1]
    # read ATL03 file using HDF5 reader
    IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams = \
        is2tk.io.ATL03.read_granule(granule, ATTRIBUTES=True, VERBOSE=True)
    # verify that data is imported correctly
    assert all(gtx in IS2_atl03_mds.keys() for gtx in IS2_atl03_beams)
    # number of GPS seconds between the GPS epoch
    # and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = IS2_atl03_mds['ancillary_data']['atlas_sdp_gps_epoch']
    # for each beam
    for gtx in IS2_atl03_beams:
        # read ICESat-2 delta time and latitude
        nref = len(IS2_atl03_mds[gtx]['geolocation']['segment_id'])
        delta_time = IS2_atl03_mds[gtx]['geophys_corr']['delta_time']
        latitude = IS2_atl03_mds[gtx]['geolocation']['reference_photon_lat']
        # read ASAS predicted long-period equilibrium tides
        fv = IS2_atl03_attrs[gtx]['geophys_corr']['tide_equilibrium']['_FillValue']
        tide_equilibrium = IS2_atl03_mds[gtx]['geophys_corr']['tide_equilibrium']
        # calculate tide time for beam
        ts = timescale.time.Timescale().from_deltatime(delta_time,
            epoch=(2018,1,1), standard='GPS')
        # calculate long-period equilibrium tides
        lpet = pyTMD.predict.equilibrium_tide(ts.tide + ts.tt_ut1, latitude)
        ii, = np.nonzero(tide_equilibrium != fv)
        # calculate differences between computed and data versions
        difference = np.ma.zeros((nref))
        difference.data[:] = lpet - tide_equilibrium
        difference.mask = (tide_equilibrium == fv)
        # will verify differences between outputs are within tolerance
        eps = np.finfo(np.float16).eps
        if not np.all(difference.mask):
            assert np.all(np.abs(difference) < eps)

# PURPOSE: Compare load pole tides with ATL03 values
@pytest.mark.skip(reason='Errors in ATL03 values?')
def test_ATL03_load_pole_tide():
    # path to local ATL03 granule
    granule = ATL03[-1]
    # read ATL03 file using HDF5 reader
    IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams = \
        is2tk.io.ATL03.read_granule(granule, ATTRIBUTES=True, VERBOSE=True)
    # verify that data is imported correctly
    assert all(gtx in IS2_atl03_mds.keys() for gtx in IS2_atl03_beams)
    # for each beam
    for gtx in IS2_atl03_beams:
        # read ICESat-2 delta time and latitude
        nref = len(IS2_atl03_mds[gtx]['geolocation']['segment_id'])
        delta_time = IS2_atl03_mds[gtx]['geophys_corr']['delta_time']
        longitude = IS2_atl03_mds[gtx]['geolocation']['reference_photon_lon']
        latitude = IS2_atl03_mds[gtx]['geolocation']['reference_photon_lat']
        # read ASAS predicted load tides
        fv = IS2_atl03_attrs[gtx]['geophys_corr']['tide_equilibrium']['_FillValue']
        tide_pole = IS2_atl03_mds[gtx]['geophys_corr']['tide_pole']
        # calculate load pole tides from correction function
        Srad = pyTMD.compute.LPT_displacements(longitude, latitude,
            delta_time, EPSG=4326, EPOCH=(2018,1,1,0,0,0), TYPE='drift',
            TIME='GPS', ELLIPSOID='IERS', CONVENTION='2003')
        # calculate differences between computed and data versions
        difference = np.ma.zeros((nref))
        difference.data[:] = Srad - tide_pole
        difference.mask = (tide_pole == fv)
        # will verify differences between outputs are within tolerance
        eps = np.finfo(np.float16).eps
        if not np.all(difference.mask):
            assert np.all(np.abs(difference) < eps)

# PURPOSE: Compare ocean load pole tides with ATL03 predicted values
@pytest.mark.skip(reason='Errors in ATL03 values?')
def test_ATL03_ocean_pole_tide():
    # path to local ATL03 granule
    granule = ATL03[-1]
    # read ATL03 file using HDF5 reader
    IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams = \
        is2tk.io.ATL03.read_granule(granule, ATTRIBUTES=True, VERBOSE=True)
    # verify that data is imported correctly
    assert all(gtx in IS2_atl03_mds.keys() for gtx in IS2_atl03_beams)
    # for each beam
    for gtx in IS2_atl03_beams:
        # read ICESat-2 delta time and latitude
        nref = len(IS2_atl03_mds[gtx]['geolocation']['segment_id'])
        delta_time = IS2_atl03_mds[gtx]['geophys_corr']['delta_time']
        longitude = IS2_atl03_mds[gtx]['geolocation']['reference_photon_lon']
        latitude = IS2_atl03_mds[gtx]['geolocation']['reference_photon_lat']
        # read ASAS predicted ocean pole load tides
        fv = IS2_atl03_attrs[gtx]['geophys_corr']['tide_equilibrium']['_FillValue']
        tide_oc_pole = IS2_atl03_mds[gtx]['geophys_corr']['tide_oc_pole']
        # calculate ocean pole tides from correction function
        Urad = pyTMD.compute.OPT_displacements(longitude, latitude,
            delta_time, EPSG=4326, EPOCH=(2018,1,1,0,0,0), TYPE='drift',
            TIME='GPS', ELLIPSOID='IERS', CONVENTION='2003')
        # calculate differences between computed and data versions
        difference = np.ma.zeros((nref))
        difference.data[:] = Urad - tide_oc_pole
        difference.mask = (tide_oc_pole == fv)
        # will verify differences between outputs are within tolerance
        eps = np.finfo(np.float16).eps
        if not np.all(difference.mask):
            assert np.all(np.abs(difference) < eps)

# path to an ATL07 file from NSIDC
ATL07 = ['https://n5eil01u.ecs.nsidc.org','ATLAS','ATL07.006','2018.10.14',
    'ATL07-01_20181014000347_02350101_006_02.h5']
# PURPOSE: Download ATL07 granule from NSIDC
@pytest.fixture(scope="module", autouse=True)
def download_ATL07(username, password):
    # only download ATL07 file if not currently existing
    if not os.access(ATL07[-1], os.F_OK):
        # download an ATL07 file from NSIDC
        is2tk.utilities.from_nsidc(ATL07, username=username,
            password=password, local=ATL07[-1], verbose=True)
        autoremove = True
    else:
        autoremove = False
    # run tests
    yield
    # remove the local file
    if autoremove:
        os.remove(ATL07[-1])

# PURPOSE: Compare long-period equilibrium tides with ATL07 values
def test_ATL07_equilibrium_tides():
    # path to local ATL07 granule
    granule = ATL07[-1]
    # read ATL07 file using HDF5 reader
    IS2_atl07_mds, IS2_atl07_attrs, IS2_atl07_beams = \
        is2tk.io.ATL07.read_granule(granule, ATTRIBUTES=True, VERBOSE=True)
    # verify that data is imported correctly
    assert all(gtx in IS2_atl07_mds.keys() for gtx in IS2_atl07_beams)
    # number of GPS seconds between the GPS epoch
    # and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = IS2_atl07_mds['ancillary_data']['atlas_sdp_gps_epoch']
    # for each beam
    for gtx in IS2_atl07_beams:
        # read ICESat-2 sea ice delta time and latitude
        nseg = len(IS2_atl07_mds[gtx]['sea_ice_segments']['height_segment_id'])
        val = IS2_atl07_mds[gtx]['sea_ice_segments']
        attrs = IS2_atl07_attrs[gtx]['sea_ice_segments']
        delta_time = val['delta_time']
        longitude = val['longitude']
        latitude = val['latitude']
        # read ASAS predicted long-period equilibrium tides
        fv = attrs['geophysical']['height_segment_lpe']['_FillValue']
        height_segment_lpe = val['geophysical']['height_segment_lpe'][:]
        # calculate tide time for beam
        ts = timescale.time.Timescale().from_deltatime(delta_time,
            epoch=(2018,1,1), standard='GPS')
        # calculate long-period equilibrium tides
        lpet = pyTMD.predict.equilibrium_tide(ts.tide + ts.tt_ut1, latitude)
        # calculate differences between computed and data versions
        difference = np.ma.zeros((nseg))
        difference.data[:] = lpet - height_segment_lpe
        difference.mask = (height_segment_lpe == fv)
        # will verify differences between outputs are within tolerance
        eps = np.finfo(np.float16).eps
        if not np.all(difference.mask):
            assert np.all(np.abs(difference) < eps)
        # calculate long-period equilibrium tides from correction function
        lpet = pyTMD.compute.LPET_elevations(longitude, latitude,
            delta_time, EPSG=4326, EPOCH=(2018,1,1,0,0,0), TYPE='drift',
            TIME='GPS')
        # calculate differences between computed and data versions
        difference = np.ma.zeros((nseg))
        difference.data[:] = lpet - height_segment_lpe
        difference.mask = (height_segment_lpe == fv)
        # will verify differences between outputs are within tolerance
        eps = np.finfo(np.float16).eps
        if not np.all(difference.mask):
            assert np.all(np.abs(difference) < eps)
