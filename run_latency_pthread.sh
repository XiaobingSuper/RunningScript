export DNNL_GRAPH_CONSTANT_CACHE=1
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export LD_PRELOAD="/localdisk/xiaobing/jemalloc/lib/libjemalloc.so /home/xiaobing/miniconda3/envs/pytorch-spr/lib/libiomp5.so"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export KMP_BLOCKTIME=1

# change this number to adjust number of instances

#BATCH_SIZE=64

export OMP_NUM_THREADS=4
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"

python_exe=/home/xiaobing/miniconda3/envs/pytorch-spr/bin/python
python_script=/localdisk/xiaobing/test/resnet50_test_pthread.py

DNNL_VERBOSE=0 numactl --physcpubind=32-63 --membind=1 $python_exe -u $python_script 2>&1 | tee pthread.log

