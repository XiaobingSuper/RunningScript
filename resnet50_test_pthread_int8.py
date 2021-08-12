import torch
import intel_pytorch_extension as ipex
import torchvision
import time
import torch.profiler as profiler
from torch.utils import ThroughputBenchmark
import threading

#torch.manual_seed(1000)
model = torchvision.models.resnet50().eval()

#model = model.to(memory_format=torch.channels_last)
#model = ipex.optimize(model, dtype=torch.bfloat16, level='O0', inplace=True)

warm_up = 100
num_iter = 300
batch_size = 1
#batch_size = 64

#x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
x =  torch.randn(batch_size, 3, 224, 224)
ipex.core.disable_jit_opt()
ipex.core._jit_set_llga_enabled(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_set_profiling_executor(True)
model = ipex.fx.conv_bn_fuse(model)
conf = ipex.AmpConf(torch.int8, "resnet50_configure.json")
with torch.no_grad(), ipex.amp.autocast(enabled=True, configure=conf):
    trace_model = torch.jit.trace(model, x, check_trace=False).eval()
trace_model = torch.jit._recursive.wrap_cpp_module(torch._C._freeze_module(trace_model._c, preserveParameters=True))
with torch.no_grad():
    y = trace_model(x)
    pp = trace_model.graph_for(x)
'''
with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
    trace_model = torch.jit.trace(model, x, check_trace=True).eval()
    #y = trace_model(x)
'''
'''
with torch.no_grad():
    y = trace_model(x)
'''

def run_model(m, tid):
    time_consume = 0
    #x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for i in range(warm_up):
            y = m(x)
        for i in range(num_iter):
            x1 = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
            x2 = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
            x3 = torch.randn(batch_size, 3, 224, 224)
            x4 = torch.randn(batch_size, 3, 224, 224)
            start_time = time.time()
            y = m(x)
            time_consume += time.time() - start_time
    print('Instance num %d Avg Time/Iteration %f msec Throughput %f images/sec' %(tid, time_consume*1000/num_iter, num_iter/time_consume))

num_instances = 14
threads = []
print("begi running.........................")
for i in range(1, num_instances+1):
    thread = threading.Thread(target=run_model, args=(trace_model, i))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

with torch.no_grad():
    bench = ThroughputBenchmark(trace_model)
    for i in range(14):
        x = torch.randn(1, 3, 224, 224)
        bench.add_input(x)

    stats = bench.benchmark(
            num_calling_threads = 14,
            num_warmup_iters = 200,
            num_iters=500 * 14)

latency = stats.latency_avg_ms
print(latency)
print(stats.iters_per_second)

