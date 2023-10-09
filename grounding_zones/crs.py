#!/usr/bin/env python
u"""
crs.py
Written by Tyler Sutterley (10/2023)

Coordinates Reference System (CRS) utilities

PYTHON DEPENDENCIES:
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/

UPDATE HISTORY:
    Written 10/2023
"""
import logging
# attempt imports
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    logging.debug("pyproj not available")

# Topex/Poseidon Ellipsoid in ITRF2008 to WGS84 Ellipsoid in ITRF2014
def tp_itrf2008_to_wgs84_itrf2014():
    """``pyproj`` transform for T/P Ellipsoid in ITRF2008 to WGS84 Ellipsoid in ITRF2014"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +a=6378136.3 +rf=298.257
        +step +proj=helmert +x=-0.0016 +y=-0.0019 +z=-0.0024 +rx=0 +ry=0 +rz=0 +s=2e-05
            +dx=0 +dy=0 +dz=0.0001 +drx=0 +dry=0 +drz=0 +ds=-3e-05
            +t_epoch=2010 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# Topex/Poseidon Ellipsoid in ITRF2008 to WGS84 Ellipsoid in ITRF2020
def tp_itrf2008_to_wgs84_itrf2020():
    """``pyproj`` transform for T/P Ellipsoid in ITRF2008 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +a=6378136.3 +rf=298.257
        +step +proj=helmert +x=-0.0002 +y=-0.001 +z=-0.0033 +rx=0 +ry=0 +rz=0 +s=0.00029
            +dx=0 +dy=0.0001 +dz=-0.0001 +drx=0 +dry=0 +drz=0 +ds=-3e-05
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF88 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf88_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF88 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=-0.0245 +y=0.0039 +z=0.1699 +rx=-0.0001 +ry=0 +rz=-0.00036 +s=-0.01147
            +dx=-0.0001 +dy=0.0006 +dz=0.0031 +drx=0 +dry=0 +drz=-2e-05 +ds=-0.00012
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF89 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf89_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF89 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=-0.0295 +y=-0.0321 +z=0.1459 +rx=0 +ry=0 +rz=-0.00036 +s=-0.00837
            +dx=-0.0001 +dy=0.0006 +dz=0.0031 +drx=0 +dry=0 +drz=-2e-05 +ds=-0.00012
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF92 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf92_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF92 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=-0.0145 +y=0.0019 +z=0.0859 +rx=0 +ry=0 +rz=-0.00036 +s=-0.00327
            +dx=-0.0001 +dy=0.0006 +dz=0.0031 +drx=0 +dry=0 +drz=-2e-05 +ds=-0.00012
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF93 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf93_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF93 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=0.0658 +y=-0.0019 +z=0.0713 +rx=0.00336 +ry=0.00433 +rz=-0.00075 +s=-0.00447
            +dx=0.0028 +dy=0.0002 +dz=0.0023 +drx=0.00011 +dry=0.00019 +drz=-7e-05 +ds=-0.00012
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF94 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf94_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF94 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=-0.0065 +y=0.0039 +z=0.0779 +rx=0 +ry=0 +rz=-0.00036 +s=-0.00398
            +dx=-0.0001 +dy=0.0006 +dz=0.0031 +drx=0 +dry=0 +drz=-2e-05 +ds=-0.00012
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF96 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf96_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF96 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=-0.0065 +y=0.0039 +z=0.0779 +rx=0 +ry=0 +rz=-0.00036 +s=-0.00398
            +dx=-0.0001 +dy=0.0006 +dz=0.0031 +drx=0 +dry=0 +drz=-2e-05 +ds=-0.00012
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF97 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf97_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF97 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=-0.0065 +y=0.0039 +z=0.0779 +rx=0 +ry=0 +rz=-0.00036 +s=-0.00398
            +dx=-0.0001 +dy=0.0006 +dz=0.0031 +drx=0 +dry=0 +drz=-2e-05 +ds=-0.00012
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF2000 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf2000_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF2000 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=0.0002 +y=-0.0008 +z=0.0342 +rx=0 +ry=0 +rz=0 +s=-0.00225
            +dx=-0.0001 +dy=0 +dz=0.0017 +drx=0 +dry=0 +drz=0 +ds=-0.00011
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF2005 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf2005_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF2005 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=-0.0027 +y=-0.0001 +z=0.0014 +rx=0 +ry=0 +rz=0 +s=-0.00065
            +dx=-0.0003 +dy=0.0001 +dz=-0.0001 +drx=0 +dry=0 +drz=0 +ds=-3e-05
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF2008 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf2008_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF2008 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=-0.0002 +y=-0.001 +z=-0.0033 +rx=0 +ry=0 +rz=0 +s=0.00029
            +dx=0 +dy=0.0001 +dz=-0.0001 +drx=0 +dry=0 +drz=0 +ds=-3e-05
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# WGS84 Ellipsoid in ITRF2014 to WGS84 Ellipsoid in ITRF2020
def wgs84_itrf2014_to_wgs84_itrf2020():
    """``pyproj`` transform for WGS84 Ellipsoid in ITRF2014 to WGS84 Ellipsoid in ITRF2020"""
    pipeline = """+proj=pipeline
        +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
        +step +proj=cart +ellps=WGS84
        +step +proj=helmert +x=0.0014 +y=0.0009 +z=-0.0014 +rx=0 +ry=0 +rz=0 +s=0.00042
            +dx=0 +dy=0.0001 +dz=-0.0002 +drx=0 +dry=0 +drz=0 +ds=0
            +t_epoch=2015 +convention=position_vector
        +step +inv +proj=cart +ellps=WGS84
        +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"""
    return pyproj.Transformer.from_pipeline(pipeline)

# PURPOSE: return WGS84 transform for a given ITRF
def get_itrf_transform(ITRF):
    """
    Get a transform for converting a given ITRF to ITRF2020

    Parameters
    ----------
    ITRF: str
        International Terrestrial Reference Frame Realization
    """
    transforms = dict(
        ITRF88=wgs84_itrf88_to_wgs84_itrf2020,
        ITRF89=wgs84_itrf89_to_wgs84_itrf2020,
        ITRF92=wgs84_itrf92_to_wgs84_itrf2020,
        ITRF93=wgs84_itrf93_to_wgs84_itrf2020,
        ITRF94=wgs84_itrf94_to_wgs84_itrf2020,
        ITRF96=wgs84_itrf96_to_wgs84_itrf2020,
        ITRF97=wgs84_itrf97_to_wgs84_itrf2020,
        ITRF2000=wgs84_itrf2000_to_wgs84_itrf2020,
        ITRF2005=wgs84_itrf2005_to_wgs84_itrf2020,
        ITRF2008=wgs84_itrf2008_to_wgs84_itrf2020,
        ITRF2014=wgs84_itrf2014_to_wgs84_itrf2020,
    )
    try:
        return transforms[ITRF]()
    except:
        logging.error(f'Invalid ITRF: {ITRF}')
        return None

# PURPOSE: return a pyproj transform direction
def get_direction(BF):
    """
    Get a pyproj transform direction

    Parameters
    ----------
    BF: str
        direction of transform (Forward or Backward)
    """
    if BF.lower() in ('f','forward',):
        return pyproj.enums.TransformDirection.FORWARD
    elif BF.lower() in ('b','backward','i','inverse',):
        return pyproj.enums.TransformDirection.INVERSE
    else:
        raise ValueError(f'Invalid direction: {BF}')
