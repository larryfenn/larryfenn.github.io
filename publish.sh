#!/bin/sh
set -e

cd _site
git add -A
git commit -m  "$1"
git push origin master
