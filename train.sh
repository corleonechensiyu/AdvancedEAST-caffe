#!/bin/sh
CAFFE_HOME=/home/csy/CAFFE_SSD/

LOG=log/east_caffe.log
SOLVER="solver.prototxt"
WEIGHTS=" "
#SNAPSHOT="mbv3_iter_89200.solverstate"
$CAFFE_HOME/build/tools/caffe train \
        --solver=$SOLVER \
		    --weights=$WEIGHTS \
        --gpu=0 2>&1 | tee $LOG
