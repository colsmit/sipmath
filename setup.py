#!/usr/bin/env python

from setuptools import setup

setup(name='sipmath',
      version='0.1',
      description='Python implementation of Sipmath Modeling tools and Metalog Distribution',
      url='http://github.com/storborg/funniest',
      author='Colin Smith, Isaac Faber',
      author_email='colin.smith.a@gmail.com',
      license='MIT',
      packages=setup.find_packages(),
      zip_safe=False,
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ]
      )


#TODO have to check if python version 3+