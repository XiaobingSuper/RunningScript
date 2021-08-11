#!/bin/sh
export DNNL_GRAPH_CONSTANT_CACHE=1
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export LD_PRELOAD="/home/xiaobing/Download/jemalloc/lib/libjemalloc.so /home/xiaobing/miniconda3/envs/pytorch-spr/lib/libiomp5.so"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
#BATCH_SIZE=64
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=56
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo -e "### using $KMP_SETTING\n\n"
sleep 3
python_exe=python
python_script=resnet50_train_test.py
#numactl --physcpubind=0-31 --membind=0 ~/miniconda3/envs/pytorch-xiaobing/bin/python -u ~/xiaobing/test/resnet50_test.py
#DNNL_VERBOSE=0 numactl --physcpubind=0-15 --membind=0 /home/sdp/miniconda3/envs/xiaobing-debug/bin/python -u /home/sdp/xiaobing/test/resnet50_test.py 2>&1 | tee verbose.log
#DNNL_VERBOSE=1 numactl --physcpubind=0-27 --membind=0 /home/sdp/miniconda3/envs/xiaobing-debug/bin/python -u /home/sdp/xiaobing/test/resnet50_test_op.py 2>&1 | tee verbose.log
#DNNL_VERBOSE=1 numactl --physcpubind=0-27 --membind=0 /home/sdp/miniconda3/envs/xiaobing-debug/bin/python -u /home/sdp/xiaobing/test/resnet50_test.py 2>&1 | tee verbose.log
DNNL_VERBOSE=0 numactl --physcpubind=0-55 --membind=0 $python_exe -u $python_script 2>&1 | tee verbose.log

