import torch
import intel_pytorch_extension as ipex
import torchvision
import time
import torch.profiler as profiler
from torch.utils import ThroughputBenchmark
import threading
import os
import torch.fx.experimental.optimization as optimization

#torch.manual_seed(1000)
model = torchvision.models.resnet50().eval()
model = ipex.fx.conv_bn_fuse(model)

warm_up = 300
num_iter = 1000
batch_size = 4

x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

conf = ipex.QuantConf("resnet50_configure.json")
trace_model = ipex.quantization.convert(model, conf, x)

with torch.no_grad():
    y = trace_model(x)
    pp = trace_model.graph_for(x)

def run_model(m, tid):
    time_consume = 0
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        for i in range(warm_up):
            y = m(x)
        for i in range(num_iter):
            #for i in range(8):
            #    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            start_time = time.time()
            y = m(x)
            time_consume += time.time() - start_time
    print('Instance num %d Avg Time/Iteration %f msec Throughput %f images/sec' %(tid, time_consume*1000/num_iter, num_iter/time_consume))

def run_model_profiler(m, tid):
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        for i in range(warm_up):
            y = m(x)
    with torch.no_grad():
        with torch.autograd.profiler.profile() as prof:
            for i in range(100):
                y = m(x)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        '''
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU],
            #schedule=torch.profiler.schedule(wait=10,warmup=50,active=10),
            # son_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_result")
            on_trace_ready=trace_handler) as p:
            for i in range(num_iter - 200):
                y = m(x)
                p.step()
        '''

num_instances = 56
threads = []
print("begin running.........................")
for i in range(1, num_instances+1):
    thread = threading.Thread(target=run_model, args=(trace_model, i))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

'''
print("cccccccccccccccccccccccccccc")
with torch.no_grad():
    bench = ThroughputBenchmark(trace_model)
    for i in range(14):
        x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        bench.add_input(x)

    stats = bench.benchmark(
            num_calling_threads = 14,
            num_warmup_iters = 200,
            num_iters=500 * 14,
            profiler_output_path="/home/xiaobing/Downloads/RunningScript")

latency = stats.latency_avg_ms
print(latency)
print(stats.iters_per_second)
'''
