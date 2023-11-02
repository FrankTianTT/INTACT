from pathlib import Path

from setuptools import find_packages, setup

def parse_requirements_file(path):
    return [line.rstrip() for line in open(path, "r")]


reqs_main = parse_requirements_file("requirements.txt")
# reqs_dev = parse_requirements_file("requirements/dev.txt")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tdfa",
    version="0.01",
    author="Honglong Tian",
    description="Target domain fast-adaptation for meta-RL with identifiable causal world model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrankTianTT/tdfa",
    packages=[package for package in find_packages() if package.startswith("tdfa")],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=reqs_main,
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
