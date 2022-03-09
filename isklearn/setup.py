#!/usr/bin/env python

from setuptools import setup

setup(name='isklearn',
	version='0.5.1',
	packages=['isklearn'],
	scripts=['isklearn/bin/isklearn-cli'],
	package_data={'isklearn': ['data/*']},
)