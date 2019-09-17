#!/usr/bin/env bash

set -e

echo "cd in $(dirname "$0")"
cd "$(dirname "$0")"
echo "Compiling protos in $(pwd)"
cd ../../..
protoc prystine_detection/prystine/protos/*.proto --python_out=.
echo 'Done'