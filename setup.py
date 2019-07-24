from setuptools import find_packages, setup

# Derive install requires from base.in first order requirements
with open("requirements/base.in") as f:
    requirements = f.read().strip()
requirements = requirements.replace("==", ">=").split()  # Convert to non-pinned for setup.py

setup(
    name="hypothesis_gufunc",
    version="0.0.1",
    packages=find_packages(),
    author_email=("ryan.turner@uber.com"),
    description="Extension to hypothesis to generate inputs for general universal (GU) numpy functions.",
    install_requires=requirements,
)