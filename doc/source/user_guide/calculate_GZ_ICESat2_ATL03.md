calculate_GZ_ICESat2_ATL03.py
=============================

- Calculates ice sheet grounding zones using ICESat-2 geolocated photon data following:
    * [Brunt et al., Annals of Glaciology, 51(55), 2010](https://doi.org/10.3189/172756410791392790)
    * [Fricker et al. Geophysical Research Letters, 33(15), 2006](https://doi.org/10.1029/2006GL026907)
    * [Fricker et al. Antarctic Science, 21(5), 2009](https://doi.org/10.1017/S095410200999023X)

### Calling Sequence
```bash
python3 calculate_GZ_ICESat2_ATL03.py --verbose input_file output_file
```
[Source code](https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/scripts/calculate_GZ_ICESat2_ATL03.py)

#### Inputs
1. `ATL03_file`: full path to ATL03 file

#### Command Line Options
- `-D X`, `--directory X`: Working data directory
- `-V`, `--verbose`: Verbose output of processing run
- `-M X`, `--mode X`: Permission mode of output file
