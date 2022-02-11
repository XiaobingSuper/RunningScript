export LD_PRELOAD="/home/xiaobing/Download/jemalloc/lib/libjemalloc.so /home/xiaobing/miniconda3/envs/pytorch-spr/lib/libiomp5.so"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

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
numactl --physcpubind=0-3 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_0_verbose.log &
numactl --physcpubind=4-7 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_1_verbose.log &
numactl --physcpubind=8-11 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_2_verbose.log &
numactl --physcpubind=12-15 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_3_verbose.log &
numactl --physcpubind=16-19 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_4_verbose.log &
numactl --physcpubind=20-23 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_5_verbose.log &
numactl --physcpubind=24-27 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_6_verbose.log &
numactl --physcpubind=28-31 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_7_verbose.log &
numactl --physcpubind=32-35 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_8_verbose.log &
numactl --physcpubind=36-39 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_9_verbose.log &
numactl --physcpubind=40-43 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_10_verbose.log &
numactl --physcpubind=44-47 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_11_verbose.log &
numactl --physcpubind=48-51 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_12_verbose.log &
numactl --physcpubind=52-55 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_13_verbose.log
EOF

#:<<EOF
python_exe=/home/sdp/miniconda3/envs/xiaobing-spr/bin/python
python_script=/home/sdp/xiaobing/RunningScript/resnet50_test.py

numactl --physcpubind=0-3 --membind=0 python -u resnet50_test.py 2>&1 | tee bs1_0_verbose.log 

#EOF
