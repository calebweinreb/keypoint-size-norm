from setuptools import setup, find_packages

print(find_packages(include=["kpsn", 'kpsn.*']))
setup(
    name='kpsn',
    version='0.0.1',
    packages=find_packages(include=["kpsn", 'kpsn.*']),
)