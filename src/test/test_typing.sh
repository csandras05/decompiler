#!/bin/bash
dir=$(dirname $(dirname $(realpath $0)))
echo "Type checking project $dir ..."
mypy --exclude '/third_party/' $dir