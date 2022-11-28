==============================
compute_tides_ICESat2_ATL07.py
==============================

- Calculates tidal elevations for correcting ICESat-2 sea ice height data
- Can use OTIS format tidal solutions provided by Ohio State University and ESR
- Can use Global Tide Model (GOT) solutions provided by Richard Ray at GSFC
- Can use Finite Element Solution (FES) models provided by AVISO

`Source code`__

.. __: https://github.com/tsutterley/Grounding-Zones/blob/main/tides/compute_tides_ICESat2_ATL07.py

.. argparse::
    :filename: compute_tides_ICESat2_ATL07.py
    :func: arguments
    :prog: compute_tides_ICESat2_ATL07.py
    :nodescription:
    :nodefault:

    --cutoff -c : @after
        * set to ``'inf'`` to extrapolate for all points
