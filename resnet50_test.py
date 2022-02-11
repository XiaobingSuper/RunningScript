import torch
import intel_extension_for_pytorch as ipex
import torchvision
import time
import torch.profiler as profiler
from torch.utils import mkldnn as mkldnn_utils
import copy
from torch.utils import ThroughputBenchmark

torch.manual_seed(1000)

#model = torchvision.models.resnet50().eval()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torchvision.models.resnet50().conv1
        self.bn = torchvision.models.resnet50().bn1
        self.relu = torchvision.models.resnet50().relu
        self.pool = torchvision.models.resnet50().maxpool
        self.layer1 = torchvision.models.resnet50().layer1
        self.layer2 = torchvision.models.resnet50().layer2
        self.layer3 = torchvision.models.resnet50().layer3
        self.layer4 = torchvision.models.resnet50().layer4
        self.pool2 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)

    def forward(self, x):
        #x = self.relu(self.bn(self.conv(x)))
        x = self.conv(x)
        #print(x.size())
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.size())
        '''
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.relu(x)
        '''
        #print(x1)
        return x

model = Model().eval()
#print(model)
model = torchvision.models.resnet50().eval()
model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.bfloat16)


warm_up = 200
batch_size = 10
#batch_size = 112

x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last).to(torch.bfloat16)

with torch.cpu.amp.autocast(), torch.no_grad():
    trace_model = torch.jit.trace(model, x).eval()
    trace_model = torch.jit.freeze(trace_model).eval()


with torch.no_grad():
    for i in range(warm_up):
        y = trace_model(x)

print("begin running...............")

num_iter = 300 
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
