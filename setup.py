import os
from setuptools import setup

# get version
with open('version.txt', mode='r', encoding='utf8') as fh:
    fallback_version = fh.read()

# list of all scripts to be included with package
scripts=[]
for s in ['scripts','subset','DAC','DEM','geoid','GZ','SL','tides']:
    scripts.extend([os.path.join(s,f) for f in os.listdir(s) if f.endswith('.py')])

# semantic version configuration for setuptools-scm
setup_requires = ["setuptools_scm"]
use_scm_version = {
    "relative_to": __file__,
    "local_scheme": "node-and-date",
    "version_scheme": "python-simplified-semver",
    "fallback_version":fallback_version,
}

setup(
    name='grounding-zones',
    use_scm_version=use_scm_version,
    scripts=scripts,
)
