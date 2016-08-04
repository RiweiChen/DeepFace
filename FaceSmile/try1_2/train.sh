#!/usr/bin/env sh
TOOLS=/media/crw/MyBook/Caffe/caffe-triplet/build/tools
GLOG_logtostderr=0 GLOG_log_dir=Log/ \
$TOOLS/caffe train --solver=solver.prototxt #--weights=small_maxout2__iter_1360000.caffemodel
