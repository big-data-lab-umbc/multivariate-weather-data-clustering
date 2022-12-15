from setuptools import find_packages
from setuptools import setup

with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

install_requires = set()
with open("requirements.txt") as f:
    for dep in f.read().split('\n'):
        if dep.strip() != '' and not dep.startswith('-e'):
            install_requires.add(dep)

setup(
    name="mwdc",
    version="1.5.0",
    author="Jianwu Wang, Francis Nji, Omar Faruque, Rohan Salvi, Mostafa Cham",
    author_email="",
    url="https://github.com/big-data-lab-umbc/multivariate-weather-data-clustering.git",
    install_requires=list(install_requires),
    packages=find_packages(
        exclude=("example*", "archive*", "Benchmark*")),
    long_description=long_description,

)