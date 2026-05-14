#!/usr/bin/env python

from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='hamsci_LSTID_detect',
      version='0.1',
      description='HamSCI Large Scale Traveling Ionospheric Disturbance Detection',
      author='Nathaniel A. Frissell',
      author_email='nathaniel.frissell@scranton.edu',
      url='https://hamsci.org',
      packages=['scripts'],
      install_requires=requirements
     )
