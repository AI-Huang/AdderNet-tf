from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [l for l in f.read().splitlines() if l]

setup(
    name="addernet",
    author="Kan HUANG",
    # install_requires=requirements,
    python_requires=">=3.6",
    version="0.0.1",
    packages=find_packages()
)
