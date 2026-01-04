#!/usr/bin/env python3

# A benchmark script to compare CPU and CUDA performance using PyTorch.
# prerequisites:
# - use a venv: python3 -m venv .venv && source .venv/bin/activate
# - CPU-only (simple): pip3 install torch tqdm
# - GPU (example for CUDA 11.8): pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# usage: python cuda.py --size 5000 --iters 20

import time
import argparse
from tqdm import tqdm
import torch

def run_benchmark(device: torch.device, size: int, iterations: int) -> float:
    if device.type == 'cuda':
        print('running on CUDA')
        print('CUDA Device Name:', torch.cuda.get_device_name(0))
        print('Number of GPUs:', torch.cuda.device_count())
    else:
        print('running on CPU')
        print('Threads:', torch.get_num_threads())
        # print('Running on CPU with threads =', torch.get_num_threads())

    # print an empty line, prevent tqdm from overwriting the previous line
    print()

    # warm up
    if device.type == 'cuda':
        torch.rand(1, device=device)
        torch.cuda.synchronize()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in tqdm(range(iterations), desc=f'{device.type.upper()} bench', unit='iter'):
        x = torch.rand(size, size, device=device)
        y = torch.rand(size, size, device=device)
        _ = x.mul(y)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f'{device.type.upper()} elapsed: {elapsed:.4f}s (avg {elapsed/iterations:.6f}s per iter)')
    print('==================================')
    return elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=5000, help='matrix size (NxN)')
    parser.add_argument('--iters', type=int, default=20, help='number of iterations')
    args = parser.parse_args()

    # print args
    print('Matrix size:', args.size)
    print('Iterations:', args.iters)
    print('==================================')

    cpu_device = torch.device('cpu')
    cuda_available = torch.cuda.is_available()
    devices = [cpu_device]
    if cuda_available:
        devices.append(torch.device('cuda'))

    results = {}
    for dev in devices:
        results[dev.type] = run_benchmark(dev, args.size, args.iters)

    if 'cpu' in results and 'cuda' in results:
        cpu_t = results['cpu']
        cuda_t = results['cuda']
        speedup = cpu_t / cuda_t if cuda_t > 0 else float('inf')
        print(f'Speedup (CPU / CUDA) = {speedup:.2f}x')

if __name__ == '__main__':
    main()
