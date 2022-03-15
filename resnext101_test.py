import torch
import intel_extension_for_pytorch as ipex
import torchvision
import time
import torch.profiler as profiler
from torch.utils import mkldnn as mkldnn_utils
import copy
from torch.utils import ThroughputBenchmark

torch.manual_seed(1000)

torch.set_flush_denormal(True)

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl').eval()

model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.bfloat16)


warm_up = 100
batch_size = 1
#batch_size = 112

x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)

with torch.cpu.amp.autocast(), torch.no_grad():
    trace_model = torch.jit.trace(model, x).eval()
    trace_model = torch.jit.freeze(trace_model).eval()


with torch.no_grad():
    for i in range(warm_up):
        y = trace_model(x)

print("begin running...............")

num_iter = 200 
fwd = 0

with torch.no_grad():
    t1  = time.time()
    for i in range(num_iter):
        y = trace_model(x)
    t2 = time.time()
    fwd = fwd + (t2 - t1)

avg_time = fwd / num_iter * 1000
print("batch_size = %d, avg time is %0.3f (ms) fps:%f"%(batch_size, avg_time, batch_size  * num_iter / fwd))
'''
num_instance=14
with torch.no_grad():
    y = trace_model(x)
bench = ThroughputBenchmark(trace_model)
for j in range(num_instance):
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
    bench.add_input(x)

stats = bench.benchmark(
        num_calling_threads=num_instance,
        num_warmup_iters=500,
        num_iters=1000 * num_instance)
print(stats)
latency = stats.latency_avg_ms
print(latency)
'''
'''
def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
    # prof.export_chrome_trace("rn50_trace_" + str(prof.step_num) + ".json")

with torch.no_grad():
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=10,warmup=50,active=10),
        # son_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_result")
        on_trace_ready=trace_handler) as p:
        for i in range(num_iter - 200):
            y = trace_model(x)
            p.step()
'''
