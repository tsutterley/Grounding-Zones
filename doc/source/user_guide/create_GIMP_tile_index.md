create_GIMP_tile_index.py
=========================

 - Reads GIMP 30m DEM tiles from the OSU [Greenland Ice Mapping Project](https://nsidc.org/data/nsidc-0645/versions/1)
 - Creates a single shapefile with the extents of each tile

#### Calling Sequence
```bash
python3 create_GIMP_tile_index.py --user <username> --directory <outgoing> \
	--release 1.1 --mode 0o775
```
[Source code](https://github.com/tsutterley/ICESat-2-Grounding-Zones/blob/main/scripts/create_GIMP_tile_index.py)

#### Command Line Options
 - `-U X`, `--user X`: username for NASA Earthdata Login
 - `-N X`, `--netrc X`: path to .netrc file for alternative authentication
 - `-D X`, `--directory`: working data directory for output GIMP files
 - `-v X`, `--version X:` data release of the GIMP dataset
 - `-M X`, `--mode X`: Local permissions mode of the directories and files synced
