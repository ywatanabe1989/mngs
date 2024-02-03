#!/bin/bash

rm -rf build dist/* src/mngs.egg-info
python3 setup.py sdist bdist_wheel
twine upload -r pypi dist/*

## EOF
