#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(name='ardour2mid', 
    version='0.1', 
    description='Convert an ardour project to an AGB midi.',
    author='Wodka',
    packages = find_packages(),
    scripts = [
        'ardour2mid.py'
        ],
    install_requires = ['numpy', 'mido', 'argparse'],
    include_package_data=True
)