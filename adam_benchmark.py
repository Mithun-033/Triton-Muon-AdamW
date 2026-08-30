import torch
import torch.nn as nn
from triton_adam import TritonAdamW
import argparse

parser = argparse.ArgumentParser(description='Benchmark Triton AdamW vs PyTorch AdamW')
parser.add_argument('--fused', action='store_true', help='Use fused AdamW in PyTorch')
args = parser.parse_args()

model=nn.Sequential(
    nn.Linear(1024,4096),
    nn.LayerNorm(4096),
    nn.ReLU(),
    nn.Linear(4096,4096),
    nn.LayerNorm(4096),
    nn.ReLU(),
    nn.Linear(4096,4096),
    nn.LayerNorm(4096),
    nn.ReLU(),
    nn.Linear(4096,1024),
).cuda()

X = torch.randn(1000, 1024).to('cuda')
Y = torch.randn(1000, 1024).to('cuda')
import copy

model_triton=model
model_pytorch=copy.deepcopy(model)

triton_optimizer=TritonAdamW(
    model_triton,
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    weight_decay=1e-4,
    eps=1e-8
)

pytorch_optimizer=torch.optim.AdamW(
    model_pytorch.parameters(),
    lr=1e-3,
    betas=(0.9,0.999),
    weight_decay=1e-4,
    eps=1e-8,
    fused = args.fused
)

def get_grads(model):
    model.zero_grad()
    output=model(X)
    loss=torch.nn.functional.mse_loss(output,Y)
    loss.backward()

get_grads(model_triton)
get_grads(model_pytorch)


def benchmark_optimizer(model,optimizer,warmup=100,iterations=500):
    # Warmup
    for _ in range(warmup):
        optimizer.step()

    torch.cuda.synchronize()

    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iterations):
        optimizer.step()

    end.record()
    torch.cuda.synchronize()

    total_ms=start.elapsed_time(end)
    ms_per_step=total_ms/iterations

    return ms_per_step

iterations = 10
triton_ms = 0
pytorch_ms = 0

for _ in range(iterations):
    triton_ms+=benchmark_optimizer(model_triton,triton_optimizer)
    pytorch_ms+=benchmark_optimizer(model_pytorch,pytorch_optimizer)

triton_ms/=iterations
pytorch_ms/=iterations

num_params=sum(p.numel() for p in model_triton.parameters())

flops_per_param=15
total_flops=num_params*flops_per_param

triton_gflops=total_flops/(triton_ms*1e6)
pytorch_gflops=total_flops/(pytorch_ms*1e6)

print(f"Parameters:       {num_params:,}")
print()
print(f"Triton AdamW:")
print(f"  Time:           {triton_ms:.4f} ms")
print(f"  GFLOP/s:        {triton_gflops:.2f}")
print()
print(f"PyTorch AdamW:")
print(f"  Time:           {pytorch_ms:.4f} ms")
print(f"  GFLOP/s:        {pytorch_gflops:.2f}")
print()
print(f"Speedup:          {pytorch_ms/triton_ms:.2f}x")

