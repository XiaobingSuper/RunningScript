import torch
import intel_extension_for_pytorch as ipex
import torchvision
import time
import torch.profiler as profiler
from torch.utils import ThroughputBenchmark
import threading

#torch.manual_seed(1000)
model = torchvision.models.resnet50().eval()

#model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.float32, level='O0', inplace=True)

warm_up = 10
num_iter = 50
batch_size = 1
#batch_size = 64

x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

with torch.no_grad():
    trace_model = torch.jit.trace(model, x).eval()
    trace_model = torch.jit.freeze(trace_model).eval()

with torch.no_grad():
    y = trace_model(x)

def run_model(m, tid):
    time_consume = 0
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        for i in range(warm_up):
            y = m(x)
        for i in range(num_iter):
            start_time = time.time()
            if tid == 1 and i % 10 == 0:
                torch._C._start_recording_time_event()
            y = m(x)
            if tid == 1 and i % 10 == 0:
                torch._C._output_time_event()
                torch._C._stop_recording_time_event()
            time_consume += time.time() - start_time
    print('Instance num %d Avg Time/Iteration %f msec Throughput %f images/sec' %(tid, time_consume*1000/num_iter, num_iter/time_consume))

num_instances = 8
threads = []
print("begi running.........................")
for i in range(1, num_instances+1):
    thread = threading.Thread(target=run_model, args=(trace_model, i))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

'''
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
'''
