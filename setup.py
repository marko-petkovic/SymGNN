from setuptools import setup, find_packages

setup(
    name="symgraph",
    version="1.0.0",
    packages=find_packages(include=["symgraph", "symgraph.*"]),
)