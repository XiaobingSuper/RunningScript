import torch
import intel_pytorch_extension as ipex
import torchvision
import time
import torch.profiler as profiler
from torch.utils import ThroughputBenchmark
import threading

torch.manual_seed(1000)
model = torchvision.models.resnet50().eval()

model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.bfloat16, level='O0', inplace=True)

warm_up = 1000
num_iter = 1000
batch_size = 1
#batch_size = 64

x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
    trace_model = torch.jit.trace(model, x, check_trace=False).eval()
    y = trace_model(x)

with torch.no_grad():
    y = trace_model(x)

def run_model(m, tid):
    time_consume = 0
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)
    for i in range(warm_up):
        y = m(x)
    start_time = time.time()
    for i in range(num_iter):
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

'''
t1 = threading.Thread(target=run_model, args=(x, trace_model, 1))
t2 = threading.Thread(target=run_model, args=(x, trace_model, 2))
t3 = threading.Thread(target=run_model, args=(x, trace_model, 3))
t4 = threading.Thread(target=run_model, args=(x, trace_model, 4))
t5 = threading.Thread(target=run_model, args=(x, trace_model, 5))
t6 = threading.Thread(target=run_model, args=(x, trace_model, 6))
t7 = threading.Thread(target=run_model, args=(x, trace_model, 7))
t8 = threading.Thread(target=run_model, args=(x, trace_model, 8))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()
'''
