#!/bin/bash

set -ex
set -o pipefail

# Set conda paths
export CONDA_PATH=./tmp/conda
export CONDA_ENVS=env
# Only check dep consistency under py3.6 due to ambiguity of importlib in reqs for >= 3.7
PY_VERSIONS_DEPS=( "3.6" )
PY_VERSIONS_TESTS=( "3.6" "3.7" "3.8" )

# Sometime pip PIP_REQUIRE_VIRTUALENV has issues with conda
export PIP_REQUIRE_VIRTUALENV=false

# Handy to know what we are working with
git --version

# Cleanup workspace, src for any old -e installs
git clean -x -f -d
rm -rf src/

# Install miniconda
if command -v conda 2>/dev/null; then
    echo "Conda already installed"
else
    # We need to use miniconda since we can't figure out ho to install py3.6 in
    # this env image. We could also use Miniconda3-latest-Linux-x86_64.sh but
    # pinning version to make reprodicible.
    echo "Installing miniconda"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # In future let's also try, for reprodicibility:
        # curl -L -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.12-MacOSX-x86_64.sh;
        curl -L -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh;
    else
        # In future let's also try, for reprodicibility:
        # curl -L -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh;
        curl -L -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh;
    fi
    chmod +x ./miniconda.sh
    ./miniconda.sh -b -p $CONDA_PATH
    rm ./miniconda.sh
fi
export PATH=$CONDA_PATH/bin:$PATH

# Setup env just for installing pre-commit to run hooks on all files
rm -rf "$CONDA_ENVS"
ENV_PATH="${CONDA_ENVS}/gufunc_commit_hooks"
conda create -y -q -p $ENV_PATH python=3.6
echo $ENV_PATH
source activate $ENV_PATH
python --version
pip freeze | sort
pip install -r requirements/tools.txt
# Now run hooks on all files, don't need to install hooks since run directly
pre-commit run --all-files
# Now can leave env with  pre-commit
conda deactivate
# Also check no changes to files by hooks
test -z "$(git diff)"
# clean up for good measure, but need to keep miniconda tmp folder
git clean -x -f -d --exclude=tmp

# Tool to get compare only the package names in pip file, delete out name of this package hypothesis-gufunc
# On mac, sed -r needs to be seed -E
nameonly () { grep -i '^[a-z0-9]' | sed -E "s/([^=]*)==.*/\1/g" | tr _ - | sed -E /^hypothesis-gufunc$/d | sort -f; }
pipcheck () { cat $@ | grep -i '^[a-z0-9]' | awk '{print $1}' | sed -E /^certifi==/d | sort -f | uniq >ask.log && pip freeze | sed -E /^certifi==/d | sort -f >got.log && diff -i ask.log got.log; }

# Set up environments for all Python versions and loop over them
rm -rf "$CONDA_ENVS"
for i in "${PY_VERSIONS_DEPS[@]}"
do
    # Now test the deps
    ENV_PATH="${CONDA_ENVS}/deps_test"
    conda create -y -q -p $ENV_PATH python=$i
    echo $ENV_PATH
    source activate $ENV_PATH
    python --version
    pip freeze | sort

    # Install all requirements, make sure they are mutually compatible
    pip install -r requirements/base.txt
    pipcheck requirements/base.txt
    pip install -r requirements/extra.txt
    pipcheck requirements/base.txt requirements/extra.txt
    pip install -r requirements/test.txt
    pipcheck requirements/base.txt requirements/extra.txt requirements/test.txt
    pip install -r requirements/docs.txt
    pipcheck requirements/base.txt requirements/extra.txt requirements/test.txt requirements/docs.txt
    # Install pipreqs and pip-compile
    pip install -r requirements/tools.txt
    pipcheck requirements/base.txt requirements/extra.txt requirements/test.txt requirements/docs.txt requirements/tools.txt

    # Install package
    python setup.py install
    pipcheck requirements/*.txt

    # Make sure .in file corresponds to what is imported
    nameonly <requirements/base.in >ask.log
    pipreqs hypothesis_gufunc/ --ignore hypothesis_gufunc/extra/ --savepath requirements_chk.in
    nameonly <requirements_chk.in >got.log
    diff ask.log got.log

    nameonly <requirements/extra.in >ask.log
    pipreqs hypothesis_gufunc/extra/ --savepath requirements_chk.in
    nameonly <requirements_chk.in >got.log
    diff ask.log got.log

    nameonly <requirements/test.in >ask.log
    pipreqs test/ --savepath requirements_chk.in
    nameonly <requirements_chk.in >got.log
    diff ask.log got.log

    nameonly <requirements/docs.in >ask.log
    pipreqs docs/ --savepath requirements_chk.in
    nameonly <requirements_chk.in >got.log
    diff ask.log got.log

    # Make sure txt file corresponds to pip compile
    pip-compile-multi -o chk

    nameonly <requirements/base.txt >ask.log
    nameonly <requirements/base.chk >got.log
    diff ask.log got.log

    nameonly <requirements/extra.txt >ask.log
    nameonly <requirements/extra.chk >got.log
    diff ask.log got.log

    nameonly <requirements/test.txt >ask.log
    nameonly <requirements/test.chk >got.log
    diff ask.log got.log

    nameonly <requirements/docs.txt >ask.log
    nameonly <requirements/docs.chk >got.log
    diff ask.log got.log

    # Deactivate virtual environment
    conda deactivate
done

# Set up environments for all Python versions and loop over them
rm -rf "$CONDA_ENVS"
for i in "${PY_VERSIONS_TESTS[@]}"
do
    # Now test the deps
    ENV_PATH="${CONDA_ENVS}/unit_test"
    conda create -y -q -p $ENV_PATH python=$i
    echo $ENV_PATH
    source activate $ENV_PATH
    python --version
    pip freeze | sort

    pip install -r requirements/test.txt
    # for some reason ``python setup.py install`` results in path issues with conda
    pip install -e .

    pytest test/ -v -s --cov=hypothesis_gufunc --cov-report html --hypothesis-seed=0
done
