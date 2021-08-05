#!/bin/sh

export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export LD_PRELOAD="/home/sdp/xiaobing/jemalloc/lib/libjemalloc.so /home/sdp/miniconda3/envs/pytorch-xiaobing/lib/libiomp5.so"
#export LD_PRELOAD="/home/sdp/xiaobing/jemalloc/lib/libjemalloc.so"
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=$CORES

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

#BATCH_SIZE=64
export KMP_BLOCKTIME=50
#export OMP_NUM_THREADS=$CORES_PER_INSTANCE
#export OMP_NUM_THREADS=32
#export OMP_NUM_THREADS=16
export OMP_NUM_THREADS=28
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3
#numactl --physcpubind=0-31 --membind=0 ~/miniconda3/envs/pytorch-xiaobing/bin/python -u ~/xiaobing/test/resnet50_test.py
#DNNL_VERBOSE=0 numactl --physcpubind=0-15 --membind=0 /home/sdp/miniconda3/envs/xiaobing-debug/bin/python -u /home/sdp/xiaobing/test/resnet50_test.py 2>&1 | tee verbose.log
#DNNL_VERBOSE=1 numactl --physcpubind=0-27 --membind=0 /home/sdp/miniconda3/envs/xiaobing-debug/bin/python -u /home/sdp/xiaobing/test/resnet50_test_op.py 2>&1 | tee verbose.log
DNNL_VERBOSE=1 numactl --physcpubind=0-27 --membind=0 /home/sdp/miniconda3/envs/xiaobing-debug/bin/python -u /home/sdp/xiaobing/test/resnet50_test.py 2>&1 | tee verbose.log
