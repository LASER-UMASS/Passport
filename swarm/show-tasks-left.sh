#!/usr/bin/env bash
SFLAGS="-u $USER -h"
#if [[ $# -gt 0 ]]; then
#    SFLAGS+=" -n $1-evaluate-file,$1-evaluate-proof"
#fi

while
    if [[ $# -gt 0 ]]; then
        TOTAL=0
        for i in "$@"; do
            JOBS=$(squeue $SFLAGS -n $i-evaluate-file,$i-evaluate-proof)
            EXIT=$?
            if [[ $EXIT -ne 0 ]]; then
                continue
            fi
            NUM_LEFT=$(echo -n "$JOBS" | wc -l)
            ((TOTAL+=$NUM_LEFT))
            echo -n '  '$NUM_LEFT'  '
        done
        echo -n $'\r'
    else
        JOBS=$(squeue $SFLAGS)
        EXIT=$?
        if [[ $EXIT -ne 0 ]]; then
           continue
        fi
        NUM_LEFT=$(echo -n "$JOBS" | wc -l)
        TOTAL=$NUM_LEFT
        echo -n $'\r'${NUM_LEFT}'  '
    fi
    sleep 0.1
    [[ ${TOTAL} -gt 0 ]]
do true; done
echo ""
