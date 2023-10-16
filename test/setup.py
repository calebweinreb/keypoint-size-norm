from setuptools import setup, find_packages

setup(
    name='kpsn_test',
    version='0.0.1',
    packages=find_packages(include=["kpsn_test", 'kpsn_test.*']),
)