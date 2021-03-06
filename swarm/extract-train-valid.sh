#!/usr/bin/env bash

TT_DIR=$HOME/work/Passport

PS=$TT_DIR/projs_split.json
NUM_PROJS=$(( $(jq ".projs_train | length" $PS) + $(jq ".projs_valid | length" $PS) ))
DEST=./proof_steps/
REALDEST=Passport/${DEST}
mkdir -p output/extract/

if [ -d "$REALDEST" ]; then
    read -r -p "Destination directory $REALDEST exists. Remove it? [y/N] " input
    case $input in
        [yY][eE][sS]|[yY])
        rm -r "$REALDEST" ;;
        *)
        echo "Aborting..." && exit 1 ;;
    esac
fi

mkdir $REALDEST
echo "$@" > ${REALDEST}/flags.txt
git log -20 > ${REALDEST}/glog.txt
git status > ${REALDEST}/gstatus.txt
git diff > ${REALDEST}/gdiff.txt
echo "CACHED" >> ${REALDEST}/gdiff.txt
git diff --cached >> ${REALDEST}/gdiff.txt

for proj_idx in $(eval echo "{1..$NUM_PROJS}"); do
  PROJ=$(cat <(jq -r ".projs_train[]" ${TT_DIR}/projs_split.json) \
             <(jq -r ".projs_valid[]" ${TT_DIR}/projs_split.json) \
             | awk "NR==$proj_idx")
  NUM_FILES=$(find ${TT_DIR}/data/${PROJ} -name "*.json" | wc -l)
  if [[ $NUM_FILES -eq 0 ]]; then
    continue
  fi
  sbatch --output output/extract/extract_steps_${PROJ}_%a.out -p longq --array=0-$(($NUM_FILES - 1 )) \
    $TT_DIR/swarm/extract-steps.sh $PROJ --output=${DEST} "$@"
done

$TT_DIR/swarm/show-tasks-left.sh -b
