export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export LD_PRELOAD="/home/sdp/xiaobing/jemalloc/lib//libjemalloc.so /home/sdp/miniconda3/envs/xiaobing-spr/lib/libiomp5.so"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=4

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

#BATCH_SIZE=64

export OMP_NUM_THREADS=4
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3
:<<EOF
numactl --physcpubind=0-3 --membind=0 python -u resnet50_test_B.py 2>&1 | tee bs1_0_verbose.log &
numactl --physcpubind=4-7 --membind=0 python -u resnet50_test_B.py 2>&1 | tee bs1_1_verbose.log &
numactl --physcpubind=8-11 --membind=0 python -u resnet50_test_B.py 2>&1 | tee bs1_2_verbose.log &
numactl --physcpubind=12-15 --membind=0 python -u resnet50_test_B.py 2>&1 | tee bs1_3_verbose.log &
numactl --physcpubind=16-19 --membind=0 python -u resnet50_test_B.py 2>&1 | tee bs1_4_verbose.log &
numactl --physcpubind=20-23 --membind=0 python -u resnet50_test_B.py 2>&1 | tee bs1_5_verbose.log &
numactl --physcpubind=24-27 --membind=0 python -u resnet50_test_B.py 2>&1 | tee bs1_6_verbose.log &
numactl --physcpubind=28-31 --membind=0 python -u resnet50_test_B.py 2>&1 | tee bs1_7_verbose.log
EOF


python_exe=/home/sdp/miniconda3/envs/xiaobing-spr/bin/python
python_script=/home/sdp/xiaobing/RunningScript/resnet50_test.py

DNNL_VERBOSE=0 numactl --physcpubind=56-111 --membind=1 $python_exe -u $python_script 2>&1 | tee benchmark.log

