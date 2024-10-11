from setuptools import find_packages, setup

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name="rainnow",
    version="1.0",
    description="Precipitation Nowcasting Tool",
    author="ds423@ic.ac.uk",
    packages=find_packages(),
    install_requires=requirements,
)
