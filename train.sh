#!/bin/sh
CAFFE_HOME=/home/csy/CAFFE_SSD

LOG=log/east.log
SOLVER="solver.prototxt"
#WEIGHTS="mbv3_iter_9200.caffemodel"
SNAPSHOT="saved_model/mbv3_iter_49200.solverstate"
$CAFFE_HOME/build/tools/caffe train \
        --solver=$SOLVER \
        --snapshot=$SNAPSHOT \
        --gpu=0 2>&1 | tee $LOG
