# PA4 CUDA Ray Tracer

## Description
This project implements a deterministic CUDA-based ray tracer for PA4.  
Each pixel is computed using a single primary ray with direct Lambertian
lighting and shadow testing from a single emissive sphere.

The project includes:

- CUDA GPU ray tracer using a 2D grid and 2D blocks
- Shared memory to cache scene data per block
- CPU reference ray tracer
- CPU vs GPU runtime comparison and speedup

## Files
- raytrace_cuda.cu — Main CUDA + CPU ray tracing implementation
- out.ppm — Output image generated at runtime

## How to Compile (Linux + CUDA)
```bash
nvcc -O3 -std=c++17 raytrace_cuda.cu -o raytrace_cuda
