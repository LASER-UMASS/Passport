#!/usr/bin/env bash

TT_DIR=$HOME/work/Passport/
EVAL_ID=$1
shift 1

COMMIT=$(git rev-parse --short HEAD)

OUTDIR=$TT_DIR/Passport/evaluation/${EVAL_ID}/
mkdir -p $OUTDIR
git log -20 > ${OUTDIR}/glog.txt
git status > ${OUTDIR}/gstatus.txt
git diff > ${OUTDIR}/gdiff.txt
echo "CACHED" >> ${OUTDIR}/gdiff.txt
git diff --cached >> ${OUTDIR}/gdiff.txt
echo "$@" > ${OUTDIR}/flags.txt
mkdir -p $OUTDIR/weights
cp -r $TT_DIR/Passport/runs/${EVAL_ID}/* $OUTDIR/weights
