nsidc_convert_GIMP_DEM.py
=========================

 - Reads GIMP 30m DEM tiles from the OSU [Greenland Ice Mapping Project](https://nsidc.org/data/nsidc-0645/versions/1)
 - Outputs as gzipped tar files similar to REMA and ArcticDEM tiles

#### Calling Sequence
```bash
python3 nsidc_convert_GIMP_DEM.py --user <username> --directory <outgoing> \
	--release 1.1 --mode 0o775
```
[Source code](https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/scripts/nsidc_convert_GIMP_DEM.py)

#### Command Line Options
 - `-U X`, `--user X`: username for NASA Earthdata Login
 - `-P X,` `--password X`: password for NASA Earthdata Login
 - `-N X`, `--netrc X`: path to .netrc file for alternative authentication
 - `-D X`, `--directory`: working data directory for output GIMP files
 - `-v X`, `--version X:` data release of the GIMP dataset
 - `-M X`, `--mode X`: Local permissions mode of the directories and files synced
