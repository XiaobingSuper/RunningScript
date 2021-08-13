export DNNL_GRAPH_CONSTANT_CACHE=1
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export LD_PRELOAD="/home/sdp/xiaobing/jemalloc/lib/libjemalloc.so /home/sdp/miniconda3/envs/xiaobing-spr/lib/libiomp5.so"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=4
export KMP_BLOCKTIME=1

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances

#export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
# change this number to adjust number of instances

#BATCH_SIZE=64


echo -e "### using $KMP_SETTING\n\n"

python_exe=python
python_script=resnet50_test_pthread_int8.py

DNNL_VERBOSE=0 numactl --physcpubind=0-55 --membind=0 $python_exe -u $python_script 2>&1 | tee pthread.log

