#!/bin/sh
export DNNL_GRAPH_CONSTANT_CACHE=1
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export LD_PRELOAD="/home/sdp/xiaobing/jemalloc/lib/libjemalloc.so /home/sdp/miniconda3/envs/xiaobing-spr/lib/libiomp5.so"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=$CORES

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

#BATCH_SIZE=64
export KMP_BLOCKTIME=1
#export OMP_NUM_THREADS=$CORES_PER_INSTANCE
#export OMP_NUM_THREADS=32
#export OMP_NUM_THREADS=16
export OMP_NUM_THREADS=56
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3
python_exe=python
script=resnet50_test_int8.py

DNNL_VERBOSE=0 numactl --physcpubind=56-111 --membind=1 $python_exe -u $script > int8_bs112.log
