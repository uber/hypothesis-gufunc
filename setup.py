from setuptools import find_packages, setup

# Derive install requires from base.in first order requirements
with open("requirements/base.in") as f:
    requirements = f.read().strip()
requirements = requirements.replace("==", ">=").split()  # Convert to non-pinned for setup.py

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="hypothesis_gufunc",
    version="0.0.4",
    packages=find_packages(),
    description="Extension to hypothesis to generate inputs for general universal (GU) numpy functions.",
    install_requires=requirements,
    url="https://github.com/uber/hypothesis-gufunc",
    project_urls={"Documentation": "https://hypothesis-gufunc.readthedocs.io"},
    author="Ryan Turner",
    author_email="rdturnermtl@github.com",
    license="Apache v2",
    long_description=long_description,
    platforms=["any"],
    classifiers=[
        "Framework :: Hypothesis",
        "Framework :: Pytest",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Testing",
    ],
)
