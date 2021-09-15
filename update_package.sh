#!/bin/bash

rm -rf build dist/* mngs.egg-info
pip uninstall mngs -y
pip install -e mngs
python3 setup.py sdist bdist_wheel
# twine upload -r testpypi dist/*
# twine upload -r pypi dist/*

# pip install --no-cache-dir --upgrade ./dist/mngs-*-py3-none-any.whl --force-reinstall

## EOF
