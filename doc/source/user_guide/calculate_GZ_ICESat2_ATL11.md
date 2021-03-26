calculate_GZ_ICESat2_ATL11.py
=============================

- Calculates ice sheet grounding zones using ICESat-2 annual land ice height data following:
    * [Brunt et al., Annals of Glaciology, 51(55), 2010](https://doi.org/10.3189/172756410791392790)
    * [Fricker et al. Geophysical Research Letters, 33(15), 2006](https://doi.org/10.1029/2006GL026907)
    * [Fricker et al. Antarctic Science, 21(5), 2009](https://doi.org/10.1017/S095410200999023X)
- Outputs an HDF5 file of flexure scaled to match the downstream tide model
- Outputs the grounding zone location, time and spatial uncertainty

### Calling Sequence
```bash
python3 calculate_GZ_ICESat2_ATL11.py --verbose input_file
```
[Source code](https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/scripts/calculate_GZ_ICESat2_ATL11.py)

#### Inputs
1. `ATL11_file`: full path to ATL11 file

#### Command Line Options
- `-D X`, `--directory X`: Working data directory for mask files
- `-T X`, `--tide X`: Tide model to use in correction
    * `CATS0201`
    * `CATS2008`
    * `TPXO9-atlas`
    * `TPXO9-atlas-v2`
    * `TPXO9-atlas-v3`
    * `TPXO9-atlas-v4`
    * `TPXO9.1`
    * `TPXO8-atlas`
    * `TPXO7.2`
    * `AODTM-5`
    * `AOTIM-5`
    * `AOTIM-5-2018`
    * `GOT4.7`
    * `GOT4.8`
    * `GOT4.10`
    * `FES2014`
-R X, --reanalysis X: Reanalysis model to run
    * [`ERA-Interim`](http://apps.ecmwf.int/datasets/data/interim-full-moda)
    * [`ERA5`](http://apps.ecmwf.int/data-catalogues/era5/?class=ea)
    * [`MERRA-2`](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)
- `-C`, `--crossovers`: Run ATL11 Crossovers
- `-P`, `--plot`: Create plots of flexural zone
- `-V`, `--verbose`: Verbose output of processing run
- `-M X`, `--mode X`: Permission mode of output file
