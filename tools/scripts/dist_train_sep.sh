#!/bin/bash
cd .. &&
python setup.py develop &&
cd tools &&
mkdir -p experiments/$1 &&
cp -r tools experiments/$1/. &&
cp -r pcdet experiments/$1/. &&
cd experiments/$1/tools &&

#pip install spconv-cu102
set -x
NGPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=$PORT \
        $(dirname "$0")/train.py --launcher pytorch ${PY_ARGS}