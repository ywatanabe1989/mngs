#!/bin/bash

rm -rf build dist/* src/mngs.egg-info
# tree src | grep \.py except for .py under __pycache__
# tree src -I __pycache__ > tree.txt
python3 setup.py sdist bdist_wheel
twine upload -r pypi dist/*

## EOF
