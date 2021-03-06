#!/usr/bin/env bash

set -e

TT_DIR=$HOME/work/Passport

[[ "$#" -lt 1 ]] && echo "Wrong number of parameters! This script takes at least one argument, a weights id" && exit 1

EVAL_ID=$1
shift 1
DEST="$TT_DIR/Passport/evaluation/$EVAL_ID"

if [ -d $DEST ]; then
    read -r -p "Destination directory $DEST exists. Remove it? [y/N] " input
    case $input in 
        [yY][eE][sS]|[yY])
        rm -r "$DEST" ;;
        *)
        read -r -p "Continue from existing run? [y/N] " input
        case $input in 
            [yY][eE][sS]|[yY])
            $TT_DIR/swarm/rerun-missing-files.py ${EVAL_ID} "$@"
            set -x
            ${TT_DIR}/swarm/show-tasks-left.sh -B 661 -s 10 ${EVAL_ID}
            set +x
            if ! ls $TT_DIR/Passport/evaluation/${EVAL_ID}/results*.json &> /dev/null; then
                echo "Evaluation failed for all files, exiting..."
                exit 1
            fi
            scancel -n ${EVAL_ID}-evaluate-file
            $TT_DIR/swarm/rerun-missing-proofs.sh -N 4000 ${EVAL_ID} "$@"
            set -x
            ${TT_DIR}/swarm/show-tasks-left.sh -b -s 20 ${EVAL_ID}
            set +x
            exit 0 ;;
            *)
            echo "Aborting..." && exit 1 ;;
        esac ;;
    esac
fi

./swarm/save-run.sh ${EVAL_ID} "$@"

set +e
for proj_idx in {0..26}; do
    $TT_DIR/swarm/evaluate-proj-parallel.sh ${EVAL_ID} $proj_idx "$@"
done

set -x
${TT_DIR}/swarm/show-tasks-left.sh -B 661 -s 20 ${EVAL_ID}
set +x
if ! ls $TT_DIR/Passport/evaluation/${EVAL_ID}/results*.json &> /dev/null; then
    echo "Evaluation failed for all files, exiting..."
    exit 1
fi
scancel -n ${EVAL_ID}-evaluate-file
${TT_DIR}/swarm/cancel-all-tasks.sh ${EVAL_ID}
${TT_DIR}/swarm/rerun-missing-proofs.sh -N 4000 ${EVAL_ID} "$@"
set -x
${TT_DIR}/swarm/show-tasks-left.sh -b -s 20 ${EVAL_ID}
set +x
