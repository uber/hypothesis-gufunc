from setuptools import find_packages, setup

# Derive install requires from base.in first order requirements
with open("requirements/base.in") as f:
    base_requirements = f.read().strip()
base_requirements = base_requirements.replace("==", ">=").split()  # Convert to non-pinned for setup.py

with open("requirements/extra.in") as f:
    extra_requirements = f.read().strip()
extra_requirements = extra_requirements.replace("==", ">=").splitlines()  # Convert to non-pinned for setup.py
extra_requirements = [pp for pp in extra_requirements if pp[0].isalnum()]

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="hypothesis_gufunc",
    version="0.0.6",
    packages=find_packages(exclude=("test/",)),
    description="Extension to hypothesis to generate inputs for general universal (GU) numpy functions.",
    python_requires=">=3.6",
    install_requires=base_requirements,
    extras_require={"xarray": extra_requirements},
    url="https://github.com/uber/hypothesis-gufunc",
    project_urls={"Documentation": "https://hypothesis-gufunc.readthedocs.io"},
    author="Ryan Turner",
    author_email="rdturnermtl@github.com",
    license="Apache v2",
    long_description=long_description,
    long_description_content_type="text/x-rst",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Testing",
    ],
)
