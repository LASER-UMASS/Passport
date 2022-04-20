#!/usr/bin/env bash

set -e

[[ "$#" -ne 1 ]] && echo "Wrong number of parameters! This script takes one argument, a project name" && exit 1

TT_DIR=$HOME/work/Passport

source ${TT_DIR}/swarm/prelude.sh

PROJ=$1
cd coq_projects
make $PROJ

