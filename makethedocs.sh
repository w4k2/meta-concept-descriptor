#!/bin/bash

cd _docs
jekyll build
cd ..
rm -rf docs/*
cp -r _docs/_site/* docs/
