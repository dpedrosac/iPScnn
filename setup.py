#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A setuptools based module for the iPScnn anaylsis """

from setuptools import setup, find_packages, sys
from os import path

here = path.abspath(path.dirname(sys.argv[0]))

# TODO 1) write a Readme.md file

setup(
    name='iPScnn',
    author='Urs Kleinholdermann, Maximilian Wullstein, David Pedrosa, University Hospital Gießen and Marburg, '
           'Philipps-University Marburg, Germany',
    version='0.9.0',
    description='Analysis of data from iPS-patients to catgorise them according to two different conditions: ON and OFF',
#    long_description=long_description,  # Optional
    url='https://github.com/dpedrosac/iPScnn',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Private',
        'Topic :: Software Development :: Data recording',
        'License :: MIT License',
        'Programming Language :: Python :: 3.7',
        ],

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=2.7,!=3.0.*,!=3.1.*',
    install_requires=['matlibplot', 'pandas', 'os', 'numpy', 'glob', 'random', 'sklearn', 'keras'],
    scripts=[
            'cnn',
            'preprocess',
           ],
    project_urls={
        'Bug Reports': 'https://github.com/dpedrosac/iPScnn/issues',
        'Funding': 'https://donate.pypi.org',
        'Say Thanks!': 'Thanks to all subjects who agreed to participate in the examination',
        'Source': 'https://github.com/dpedrosac/iPScnn/',
        },
)