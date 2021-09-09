import torch
import torchvision

import intel_pytorch_extension as ipex
import time

batch_size = 112
warm_up = 300
num_iter = 500


conf = ipex.AmpConf(torch.int8, 'resnet50_configure_xpu.json')
model = torchvision.models.resnet50().to(ipex.DEVICE).eval()

x = torch.randn(batch_size, 3, 224, 224).to(ipex.DEVICE)

with torch.no_grad():
    trace_model = torch.jit.trace(model, x)

with torch.no_grad(), ipex.AutoMixPrecision(conf, running_mode="inference"):
    for i in range(warm_up):
        y = trace_model(x)

print("begin running...............")

fwd = 0
with torch.no_grad(), ipex.AutoMixPrecision(conf, running_mode="inference"):
    t1  = time.time()
    for i in range(num_iter):
        y = trace_model(x)
    t2 = time.time()
    fwd = fwd + (t2 - t1)

avg_time = fwd / num_iter * 1000
print("batch_size = %d, running time is %0.3f (ms) fps:%f"%(batch_size, avg_time, batch_size  * num_iter / fwd))


