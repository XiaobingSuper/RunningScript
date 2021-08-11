import torch
import intel_pytorch_extension as ipex
import torchvision
import time
import torch.profiler as profiler
from torch.utils import mkldnn as mkldnn_utils
import copy
from torch.utils import ThroughputBenchmark
import torch.nn as nn

torch.manual_seed(1000)

model = torchvision.models.resnet50().train()
model = model.to(memory_format=torch.channels_last)
optimizer = torch.optim.SGD(model.parameters(), 0.01)
criterion = nn.CrossEntropyLoss()


conf = ipex.AmpConf(torch.bfloat16)
model, optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer, inplace=True)


warm_up = 30
num_iter = 300
#batch_size = 1
batch_size = 112

x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
target = torch.arange(1, batch_size + 1).long()

for i in range(warm_up):
    with ipex.amp.autocast(enabled=True, configure=conf):
        output = model(x)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print("begin running...............")
'''
num_iter = 1000
fwd = 0

with torch.no_grad():
    t1  = time.time()
    for i in range(num_iter):
        y = trace_model(x)
    t2 = time.time()
    fwd = fwd + (t2 - t1)

avg_time = fwd / num_iter * 1000
print("batch_size = %d, running time is %0.3f (ms) fps:%f"%(batch_size, avg_time, batch_size  * num_iter / fwd))
'''
def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
    # prof.export_chrome_trace("rn50_trace_" + str(prof.step_num) + ".json")

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=10,warmup=50,active=10),
    # son_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_result")
    on_trace_ready=trace_handler) as p:
    for i in range(num_iter - 200):
        with ipex.amp.autocast(enabled=True, configure=conf):
            output = model(x)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p.step()
