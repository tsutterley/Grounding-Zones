#!/usr/bin/env python
u"""
Model.py
Written by Tyler Sutterley (11/2025)
"""
import pyTMD.io

# PURPOSE: experimental extension of pyTMD.io.model for xarray I/O
class Model(pyTMD.io.model):
    """Retrieves tide model parameters for named models or
    from a model definition file for use in the pyTMD tide
    prediction programs

    Attributes
    ----------
    atl03: str
        HDF5 dataset string for output ATL03 tide heights
    atl06: str
        HDF5 dataset string for output ATL06 tide heights
    atl07: str
        HDF5 dataset string for output ATL07 tide heights
    atl10: str
        HDF5 dataset string for output ATL10 tide heights
    atl11: str
        HDF5 dataset string for output ATL11 tide heights
    atl12: str
        HDF5 dataset string for output ATL12 tide heights
    description: str
        HDF5 ``description`` attribute string for output tide heights
    gla12: str
        HDF5 dataset string for output GLA12 tide heights
    long_name: str
        HDF5 ``long_name`` attribute string for output tide heights
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def atl03(self) -> str:
        """Returns ICESat-2 ATL03 attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return 'tide_ocean'
        elif (self.z.variable == 'tide_load'):
            return 'tide_load'
        elif (self.z.variable == 'tide_lpe'):
            return 'tide_equilibrium'
        else:
            return None

    @property
    def atl06(self) -> str:
        """Returns ICESat-2 ATL06 attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return 'tide_ocean'
        elif (self.z.variable == 'tide_load'):
            return 'tide_load'
        elif (self.z.variable == 'tide_lpe'):
            return 'tide_equilibrium'
        else:
            return None

    @property
    def atl07(self) -> str:
        """Returns ICESat-2 ATL07 attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return 'height_segment_ocean'
        elif (self.z.variable == 'tide_load'):
            return 'height_segment_load'
        elif (self.z.variable == 'tide_lpe'):
            return 'height_segment_lpe'
        else:
            return None

    @property
    def atl10(self) -> str:
        """Returns ICESat-2 ATL07 attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return 'height_segment_ocean'
        elif (self.z.variable == 'tide_load'):
            return 'height_segment_load'
        elif (self.z.variable == 'tide_lpe'):
            return 'height_segment_lpe'
        else:
            return None

    @property
    def atl11(self) -> str:
        """Returns ICESat-2 ATL11 attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return 'tide_ocean'
        elif (self.z.variable == 'tide_load'):
            return 'tide_load'
        elif (self.z.variable == 'tide_lpe'):
            return 'tide_equilibrium'
        else:
            return None

    @property
    def atl12(self) -> str:
        """Returns ICESat-2 ATL12 attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return 'tide_ocean_seg'
        elif (self.z.variable == 'tide_load'):
            return 'tide_load_seg'
        elif (self.z.variable == 'tide_lpe'):
            return 'tide_equilibrium_seg'
        else:
            return None

    @property
    def gla12(self) -> str:
        """Returns ICESat GLA12 attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return 'd_ocElv'
        elif (self.z.variable == 'tide_load'):
            return 'd_ldElv'
        elif (self.z.variable == 'tide_lpe'):
            return 'd_eqElv'
        else:
            return None

    @property
    def long_name(self) -> str:
        """Returns ``long_name`` attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return 'ocean_tide_elevation'
        elif (self.z.variable == 'tide_load'):
            return 'load_tide_elevation'
        elif (self.z.variable == 'tide_lpe'):
            return 'equilibrium_tide_elevation'
        else:
            return None

    @property
    def description(self) -> str:
        """Returns ``description`` attribute string for a given variable
        """
        if (self.z.variable == 'tide_ocean'):
            return "Ocean tidal elevations derived from harmonic constants"
        elif (self.z.variable == 'tide_load'):
            return ("Local displacement due to ocean tidal loading "
                "derived from harmonic constants")
        elif (self.z.variable == 'tide_lpe'):
            return 'Long-period equilibrium tide elevations'
        else:
            return None
