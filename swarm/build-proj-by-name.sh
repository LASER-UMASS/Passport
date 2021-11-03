#!/usr/bin/env bash

set -e

[[ "$#" -ne 1 ]] && echo "Wrong number of parameters! This script takes one argument, a project name" && exit 1

# determine physical directory of this script
TT_DIR=$HOME/work/TacTok

read-opam.sh
opam switch "4.07.1+flambda"
eval $(opam env)

cd $TT_DIR
COQ_ROOT=$(pwd)/coq
COQBIN=$COQ_ROOT/bin/
export PATH=$COQBIN:$PATH

PROJ=$1
cd coq_projects
make $PROJ

