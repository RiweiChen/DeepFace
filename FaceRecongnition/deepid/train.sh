#!/usr/bin/env sh

TOOLS=./build/tools
GLOG_logtostderr=0 GLOG_log_dir=FaceRecognition/try5_2/Log/ \
$TOOLS/caffe train \
  --solver=FaceRecognition/try5_2/solver.prototxt # \
#  --snapshot=/media/crw/MyBook/Model/FaceRecognition/try5_2/snapshot_iter_1100000.solverstate

