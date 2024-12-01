import os
from setuptools import find_packages, setup

DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(DIR, "VERSION"), "r") as file:
    VERSION = file.read()

with open(os.path.join(DIR, "README.rst"), "r") as file:
    LONG_DESCRIPTION = file.read()

long_description = LONG_DESCRIPTION,

setup(
    name='rennpa',
    packages=find_packages(),
    include_package_data=True,
    version = VERSION,
    description='rennpa - REcurrent Neural Network Python API, a RNN library for satellite image time series. ',
    author='Gabriel Sansigolo',
    author_email = "gabrielsansigolo@gmail.com",
    url = "https://github.com/GSansigolo/rennpa",
    install_requires= [
    ],
    long_description = LONG_DESCRIPTION,
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
