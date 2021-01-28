calculate_grounding_zone.py
===========================

- Calculates ice sheet grounding zones following:
    * [Brunt et al., Annals of Glaciology, 51(55), 2010](https://doi.org/10.3189/172756410791392790)
    * [Fricker et al. Geophysical Research Letters, 33(15), 2006](https://doi.org/10.1029/2006GL026907)
    * [Fricker et al. Antarctic Science, 21(5), 2009](https://doi.org/10.1017/S095410200999023X)

### Calling Sequence
```bash
python3 calculate_grounding_zone.py --model <model> --format <format> --verbose input_file output_file
```
[Source code](https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/scripts/calculate_grounding_zone.py)

#### Inputs
 1. `input_file`: name of input file
 2. `output_file`: name of output file

#### Command Line Options
- `-D X`, `--directory X`: Working data directory
- `-m X`, `--model X`: Digital elevation model to run
    * `'REMA'`
    * `'ArcticDEM'`
    * `'GIMP'`
- `--format X`: input and output data format
    * `'csv'` (default)
    * `'netCDF4'`
    * `'HDF5'`
- `--variables X`: variable names of data in csv, HDF5 or netCDF4 file
    * for csv files: the order of the columns within the file
    * for HDF5 and netCDF4 files: time, y, x and data variable names
- `-H X`, `--header X`: number of header lines for csv files
- `--projection X`: spatial projection as EPSG code or PROJ4 string
    * `4326`: latitude and longitude coordinates on WGS84 reference ellipsoid
- `-V`, `--verbose`: Verbose output of processing run
- `-M X`, `--mode X`: Permission mode of output file
