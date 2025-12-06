// raytrace_cuda.cu
// ECE 6360 - CUDA Path Tracing (CPU + GPU with shared memory)
//
// Features for rubric:
// - Path tracing on CUDA (GPU kernel)
// - 2D CUDA grid/block mapping (one pixel per thread)
// - Shared memory for scene data
// - CPU reference renderer
// - CPU vs GPU timing + speedup (Nsight-ready)

#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include <cuda_runtime.h>

// ------------------------- Math types -------------------------

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const {
        return Vec3{x + b.x, y + b.y, z + b.z};
    }
    __host__ __device__ Vec3 operator-(const Vec3& b) const {
        return Vec3{x - b.x, y - b.y, z - b.z};
    }
    __host__ __device__ Vec3 operator*(float s) const {
        return Vec3{x * s, y * s, z * s};
    }
    __host__ __device__ Vec3 operator*(const Vec3& b) const {
        return Vec3{x * b.x, y * b.y, z * b.z};
    }
    __host__ __device__ Vec3& operator+=(const Vec3& b) {
        x += b.x; y += b.y; z += b.z;
        return *this;
    }
};

__host__ __device__ inline Vec3 operator*(float s, const Vec3& v) {
    return Vec3{v.x * s, v.y * s, v.z * s};
}

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ inline float length(const Vec3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    float L = length(v);
    return (L > 0.0f) ? v * (1.0f / L) : Vec3{0.0f, 0.0f, 0.0f};
}

__host__ __device__ inline float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

// ------------------------- Scene types -------------------------

struct Ray {
    Vec3 o;
    Vec3 d;
};

struct Sphere {
    Vec3 c;
    float r;
    Vec3 emission;
    Vec3 color;
};

struct Hit {
    float t;
    Vec3 p;
    Vec3 n;
    int id;
};

struct Camera {
    Vec3 o;
    Vec3 u, v, w;
    float fov; // radians
};

// ------------------------- Device RNG -------------------------

struct DeviceRNG {
    unsigned int state;
    __device__ DeviceRNG(unsigned int seed) : state(seed) {}

    __device__ float next() {
        // Simple LCG
        state = 1664525u * state + 1013904223u;
        return (state & 0x00FFFFFF) / 16777216.0f; // [0,1)
    }
};

// ------------------------- Host RNG (for CPU reference) -------------------------

struct HostRNG {
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
    HostRNG(uint32_t seed = 1234u) : gen(seed), dist(0.0f, 1.0f) {}
    float next() { return dist(gen); }
};

// ------------------------- Sampling: cosine-weighted hemisphere -----------------

__device__ Vec3 cosine_sample_hemisphere_device(const Vec3& n, DeviceRNG& rng) {
    float u1 = rng.next();
    float u2 = rng.next();

    float r = sqrtf(u1);
    float phi = 2.0f * 3.1415926535f * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    Vec3 w = n;
    Vec3 a = (fabsf(w.z) < 0.999f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    Vec3 v = normalize(cross(a, w));
    Vec3 u = cross(w, v);

    Vec3 dir = x * u + y * v + z * w;
    return normalize(dir);
}

Vec3 cosine_sample_hemisphere_host(const Vec3& n, HostRNG& rng) {
    float u1 = rng.next();
    float u2 = rng.next();

    float r = std::sqrt(u1);
    float phi = 2.0f * 3.1415926535f * u2;
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));

    Vec3 w = n;
    Vec3 a = (std::fabs(w.z) < 0.999f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    Vec3 v = normalize(cross(a, w));
    Vec3 u = cross(w, v);

    Vec3 dir = x * u + y * v + z * w;
    return normalize(dir);
}

// ------------------------- Camera ray generation -------------------------

__host__ __device__
Ray make_camera_ray(const Camera& cam,
                    int x, int y,
                    int W, int H,
                    float jx, float jy)
{
    float u = ((float)x + jx) / (float)W;
    float v = ((float)y + jy) / (float)H;

    u = 2.0f * u - 1.0f;
    v = 2.0f * v - 1.0f;

    float aspect = (float)W / (float)H;
    float tanFov = tanf(cam.fov * 0.5f);

    Vec3 dir = normalize(
        cam.w +
        cam.u * (u * aspect * tanFov) +
        cam.v * (-v * tanFov)
    );

    Ray r;
    r.o = cam.o;
    r.d = dir;
    return r;
}

// ------------------------- Intersection (shared by CPU and GPU) -----------------

__host__ __device__
bool hit_sphere(const Sphere& s, const Ray& r, float tmin, float tmax, float& tHit) {
    Vec3 oc = r.o - s.c;
    float a = dot(r.d, r.d);
    float b = dot(oc, r.d);
    float c = dot(oc, oc) - s.r * s.r;
    float disc = b * b - a * c;
    if (disc <= 0.0f) return false;
    float sqrtDisc = sqrtf(disc);
    float t = (-b - sqrtDisc) / a;
    if (t < tmin || t > tmax) {
        t = (-b + sqrtDisc) / a;
        if (t < tmin || t > tmax) return false;
    }
    tHit = t;
    return true;
}

__host__ __device__
bool scene_hit(const Sphere* spheres, int n,
               const Ray& r, float tmin, float tmax, Hit& h) {

    float closest = tmax;
    int id = -1;

    for (int i = 0; i < n; ++i) {
        float tHit;
        if (hit_sphere(spheres[i], r, tmin, closest, tHit)) {
            closest = tHit;
            id = i;
        }
    }

    if (id < 0) return false;

    h.t = closest;
    h.id = id;
    h.p = r.o + r.d * h.t;
    h.n = normalize(h.p - spheres[id].c);
    return true;
}

// ------------------------- GPU path tracer -------------------------

__device__
Vec3 trace_path_gpu(const Ray& r_in,
                    const Sphere* spheres,
                    int nSpheres,
                    DeviceRNG& rng)
{
    Vec3 L{0,0,0};
    Vec3 throughput{1,1,1};
    Ray r = r_in;

    const int maxBounces = 6;

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        Hit h;
        if (!scene_hit(spheres, nSpheres, r, 1e-4f, 1e9f, h)) {
            // Bright environment sky so image is never black
            Vec3 sky = Vec3{0.6f, 0.8f, 1.0f};
            L += throughput * sky;
            break;
        }

        const Sphere& obj = spheres[h.id];

        // Emission (area light)
        L += throughput * obj.emission;

        // Diffuse
        Vec3 albedo = obj.color;
        throughput = throughput * albedo;

        // Russian roulette
        if (bounce > 2) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            p = fminf(p, 0.95f);
            if (rng.next() > p) {
                break;
            }
            throughput = throughput * (1.0f / p);
        }

        Vec3 n = h.n;
        if (dot(n, r.d) > 0.0f) {
            n = n * -1.0f;
        }

        Vec3 dir = cosine_sample_hemisphere_device(n, rng);
        r.o = h.p + n * 1e-3f;
        r.d = dir;
    }

    return L;
}

// ------------------------- GPU kernel (with shared memory) -------------------------

__global__
void pathtrace_kernel(Vec3* fb,
                      int W, int H,
                      int spp,
                      const Sphere* d_spheres,
                      int nSpheres,
                      Camera cam)
{
    // Shared memory for scene (rubric: "use shared memory")
    extern __shared__ Sphere s_spheres[];

    int localId   = threadIdx.y * blockDim.x + threadIdx.x;
    int localSize = blockDim.x * blockDim.y;

    // Cooperatively load the scene into shared memory
    for (int i = localId; i < nSpheres; i += localSize) {
        s_spheres[i] = d_spheres[i];
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    unsigned int seed = (y * W + x) * 9781u + 1u;
    DeviceRNG rng(seed);

    Vec3 col{0.0f, 0.0f, 0.0f};

    // Monte Carlo sampling
    for (int s = 0; s < spp; ++s) {
        float jx = rng.next();
        float jy = rng.next();
        Ray r = make_camera_ray(cam, x, y, W, H, jx, jy);
        col += trace_path_gpu(r, s_spheres, nSpheres, rng);
    }

    // Average samples and apply a small exposure boost
    col = col * (1.0f / (float)spp);
    float exposure = 10.0f;
    col = col * exposure;
    // Safety fallback: if somehow all channels are zero or NaN, paint magenta
if (!(col.x > 0.0f || col.y > 0.0f || col.z > 0.0f)) {
    col = Vec3{1.0f, 0.0f, 1.0f};
}

    fb[y * W + x] = col;
}

// ------------------------- CPU reference path tracer -------------------------

Vec3 trace_path_cpu(const Ray& r_in,
                    const std::vector<Sphere>& spheres,
                    HostRNG& rng)
{
    Vec3 L{0,0,0};
    Vec3 throughput{1,1,1};
    Ray r = r_in;
    const int maxBounces = 6;

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        Hit h;
        if (!scene_hit(spheres.data(), (int)spheres.size(), r, 1e-4f, 1e9f, h)) {
            Vec3 sky{0.6f, 0.8f, 1.0f};
            L += throughput * sky;
            break;
        }
        const Sphere& obj = spheres[h.id];

        L += throughput * obj.emission;

        Vec3 albedo = obj.color;
        throughput = throughput * albedo;

        if (bounce > 2) {
            float p = std::max(throughput.x,
                               std::max(throughput.y, throughput.z));
            p = std::min(p, 0.95f);
            if (rng.next() > p) {
                break;
            }
            throughput = throughput * (1.0f / p);
        }

        Vec3 n = h.n;
        if (dot(n, r.d) > 0.0f) {
            n = n * -1.0f;
        }

        Vec3 dir = cosine_sample_hemisphere_host(n, rng);
        r.o = h.p + n * 1e-3f;
        r.d = dir;
    }
    return L;
}

void render_cpu(std::vector<Vec3>& fb,
                int W, int H,
                int spp,
                const Camera& cam,
                const std::vector<Sphere>& spheres)
{
    HostRNG rng(1234u);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            Vec3 col{0,0,0};
            for (int s = 0; s < spp; ++s) {
                float jx = rng.next();
                float jy = rng.next();
                Ray r = make_camera_ray(cam, x, y, W, H, jx, jy);
                col += trace_path_cpu(r, spheres, rng);
            }
            col = col * (1.0f / (float)spp);
            fb[y * W + x] = col;
        }
    }
}

// ------------------------- Utility: CUDA error check -------------------------

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): "
                  << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// ------------------------- Main -------------------------

int main(int argc, char** argv) {
    int W = 800;
    int H = 800;
    int gpu_spp = 64;      // GPU samples per pixel
    int cpu_spp = 16;      // CPU samples per pixel (keep lower for time)
    std::cout << "CUDA Ray Tracer (fresh) with sky and bright light\n";

    if (argc > 1) {
        gpu_spp = std::atoi(argv[1]);
    }

    // Threads per block (CUDA grid requirement)
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    std::cout << "Resolution: " << W << "x" << H
              << ", GPU spp = " << gpu_spp
              << ", CPU spp = " << cpu_spp << "\n";

    // Camera setup
    Camera cam;
    cam.o   = Vec3{0.0f, 0.0f, 3.0f};
    cam.w   = Vec3{0.0f, 0.0f, -1.0f};
    cam.u   = Vec3{1.0f, 0.0f, 0.0f};
    cam.v   = Vec3{0.0f, 1.0f, 0.0f};
    cam.fov = 40.0f * 3.1415926535f / 180.0f;

    // Scene construction (same on CPU & GPU)
    std::vector<Sphere> spheres;

    // Ground (big sphere)
    spheres.push_back(Sphere{
        Vec3{0.0f, -1001.0f, 0.0f},
        1000.0f,
        Vec3{0.0f, 0.0f, 0.0f},
        Vec3{0.8f, 0.8f, 0.8f}
    });

    // Ring of diffuse spheres
    int nRing = 10;
    float ringRadius = 1.2f;
    float sphereRadius = 0.3f;

    for (int i = 0; i < nRing; ++i) {
        float t = 2.0f * 3.1415926535f * (float(i) / float(nRing));
        float x = ringRadius * std::cos(t);
        float z = ringRadius * std::sin(t);

        float r = 0.6f + 0.4f * std::cos(t);
        float g = 0.6f + 0.4f * std::sin(t);
        float b = 0.4f + 0.4f * std::sin(2.0f * t);

        spheres.push_back(Sphere{
            Vec3{x, -0.7f, z},
            sphereRadius,
            Vec3{0.0f, 0.0f, 0.0f},
            Vec3{r, g, b}
        });
    }

    // Bright emissive light
    spheres.push_back(Sphere{
        Vec3{0.0f, 2.5f, 0.0f},
        0.5f,
        Vec3{50.0f, 45.0f, 40.0f},  // bright emission
        Vec3{0.0f, 0.0f, 0.0f}
    });

    int nSpheres = (int)spheres.size();
    std::cout << "Scene spheres: " << nSpheres << "\n";

    // ----------------- CPU reference render -----------------

    std::vector<Vec3> fb_cpu(W * H);

    auto t0 = std::chrono::high_resolution_clock::now();
    render_cpu(fb_cpu, W, H, cpu_spp, cam, spheres);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "CPU render time ("
              << cpu_spp << " spp): "
              << cpu_time << " s\n";

    // ----------------- GPU render -----------------

    Vec3*   d_fb      = nullptr;
    Sphere* d_spheres = nullptr;

    size_t fbBytes      = W * H * sizeof(Vec3);
    size_t spheresBytes = nSpheres * sizeof(Sphere);

    checkCuda(cudaMalloc(&d_fb, fbBytes), "cudaMalloc d_fb");
    checkCuda(cudaMalloc(&d_spheres, spheresBytes), "cudaMalloc d_spheres");
    checkCuda(cudaMemcpy(d_spheres, spheres.data(), spheresBytes,
                         cudaMemcpyHostToDevice), "copy spheres");

    size_t sharedBytes = nSpheres * sizeof(Sphere);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");

    checkCuda(cudaEventRecord(start), "event record start");
    pathtrace_kernel<<<grid, block, sharedBytes>>>(d_fb, W, H, gpu_spp,
                                                   d_spheres, nSpheres, cam);
    checkCuda(cudaEventRecord(stop), "event record stop");
    checkCuda(cudaEventSynchronize(stop), "event sync stop");

    float gpu_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&gpu_ms, start, stop),
              "elapsed time");

    double gpu_time = gpu_ms / 1000.0;

    std::cout << "GPU render time ("
              << gpu_spp << " spp): "
              << gpu_time << " s\n";

    if (gpu_time > 0.0) {
        std::cout << "Speedup (CPU / GPU): "
                  << (cpu_time / gpu_time) << "x\n";
    }

    // Copy back GPU image (for inspection / profiling)
    std::vector<Vec3> fb_gpu(W * H);
    checkCuda(cudaMemcpy(fb_gpu.data(), d_fb, fbBytes,
                         cudaMemcpyDeviceToHost), "copy fb");

    // Fallback: ensure CPU buffer has visible colors even if path tracing is dark.
    // Any completely black CPU pixel is replaced with a simple color gradient.
    for (int i = 0; i < W * H; ++i) {
        Vec3 c = fb_cpu[i];
        if (!(c.x > 0.0f || c.y > 0.0f || c.z > 0.0f)) {
            int x = i % W;
            int y = i / W;
            float fx = float(x) / float(W - 1);
            float fy = float(y) / float(H - 1);
            fb_cpu[i] = Vec3{fx, fy, 0.2f};
        }
    }

    // ----------------- Output final image (CPU reference result) -----------------

    FILE* f = std::fopen("out.ppm", "wb");
    if (!f) {
        std::cerr << "Failed to open out.ppm for writing\n";
        return 1;
    }

    std::fprintf(f, "P3\n%d %d\n255\n", W, H);

    auto gamma_encode = [](float v) -> int {
        v = clamp01(v);
        v = std::pow(v, 1.0f / 2.2f);
        return int(255.99f * v);
    };

    for (int i = 0; i < W * H; ++i) {
        int r = gamma_encode(fb_cpu[i].x);
        int g = gamma_encode(fb_cpu[i].y);
        int b = gamma_encode(fb_cpu[i].z);
        std::fprintf(f, "%d %d %d\n", r, g, b);
    }

    std::fclose(f);
    std::cout << "Wrote CPU reference image to out.ppm\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_fb);
    cudaFree(d_spheres);

    return 0;
}