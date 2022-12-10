from setuptools import find_packages, setup

setup(
    name="project",
    version="0.0.1",
    packages=find_packages(
        include=[
            "eda",
            "helpers",
            "modelling",
            "pipelines",
        ]
    ),
)
