// peak_flops_statistics.cu

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define FLOP_ITERATIONS 1000000
#define THREADS_PER_BLOCK 512 // 256 for V100/A100, 512 for H100
#define NUM_BLOCKS 10000  // Adjust based on your GPU's compute capability

// #define FP8

// Number of times to repeat the test
#define NUM_RUNS 10

// FP32 Kernel
__global__ void peak_flops_fp32(float *out) {
   float a = 1.0f;
   float b = 2.0f;
   float c = 3.0f;
   float d = 4.0f;

   #pragma unroll
   for (int i = 0; i < FLOP_ITERATIONS; ++i) {
      a = a * b + c;
      b = b * c + d;
      c = c * d + a;
      d = d * a + b;
   }

   out[threadIdx.x + blockIdx.x * blockDim.x] = a + b + c + d;
}

// FP64 Kernel
__global__ void peak_flops_fp64(double *out) {
   double a = 1.0;
   double b = 2.0;
   double c = 3.0;
   double d = 4.0;

   #pragma unroll
   for (int i = 0; i < FLOP_ITERATIONS; ++i) {
      a = a * b + c;
      b = b * c + d;
      c = c * d + a;
      d = d * a + b;
   }

   out[threadIdx.x + blockIdx.x * blockDim.x] = a + b + c + d;
}

// FP16 Kernel
__global__ void peak_flops_fp16(__half *out) {
   __half a = __float2half(1.0f);
   __half b = __float2half(2.0f);
   __half c = __float2half(3.0f);
   __half d = __float2half(4.0f);

   #pragma unroll
   for (int i = 0; i < FLOP_ITERATIONS; ++i) {
      a = __hadd(__hmul(a, b), c); // a = a * b + c
      b = __hadd(__hmul(b, c), d); // b = b * c + d
      c = __hadd(__hmul(c, d), a); // c = c * d + a
      d = __hadd(__hmul(d, a), b); // d = d * a + b
   }

   out[threadIdx.x + blockIdx.x * blockDim.x] = __hadd(__hadd(a, b), __hadd(c, d));
}

#ifdef FP8
#include <cuda_fp8.h>  // Header for FP8 types and functions (hypothetical)
// FP8 Kernel (Hypothetical)
__global__ void peak_flops_fp8(__nv_fp8_e4m3 *out) {
    __nv_fp8_e4m3 a(1.0f);
    __nv_fp8_e4m3 b(2.0f);
    __nv_fp8_e4m3 c(3.0f);
    __nv_fp8_e4m3 d(4.0f);

    #pragma unroll
    for (int i = 0; i < FLOP_ITERATIONS; ++i) {
        // Perform arithmetic operations
        a = a * b + c;
        b = b * c + d;
        c = c * d + a;
        d = d * a + b;
    }

    // Store the result to prevent optimization
    out[threadIdx.x + blockIdx.x * blockDim.x] = a + b + c + d;
}
#endif

void calculate_statistics(double results[], int num_runs, double *mean, double *stddev) {
   double sum = 0.0;
   for (int i = 0; i < num_runs; ++i) {
      sum += results[i];
   }
   *mean = sum / num_runs;

   double variance = 0.0;
   for (int i = 0; i < num_runs; ++i) {
      variance += (results[i] - *mean) * (results[i] - *mean);
   }
   variance /= num_runs;
   *stddev = sqrt(variance);
}

int main() {

   // Get the device ID
    int device_id;
    cudaGetDevice(&device_id);

    // Get device properties
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    // Print GPU name and compute capability
    printf("GPU Device:         %s\n", device_prop.name);
    printf("Compute Capability: %d.%d\n", device_prop.major, device_prop.minor);

   
   // print the settings for this test and the name/type of the GPU used
   printf("FLOP_ITERATIONS:     %d\n", FLOP_ITERATIONS);
   printf("THREADS_PER_BLOCK:   %d\n", THREADS_PER_BLOCK);
   printf("NUM_BLOCKS:          %d\n", NUM_BLOCKS);
   printf("NUM_RUNS:            %d\n", NUM_RUNS);


   // Measure FP64 Performance
   {
      // printf("Measuring FP64 peak FLOPS...\n");

      double *d_out;
      cudaMalloc((void **)&d_out, THREADS_PER_BLOCK * NUM_BLOCKS * sizeof(double));

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      double gflops_results[NUM_RUNS];

      // Warm-up run
      peak_flops_fp64<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);
      cudaDeviceSynchronize();

      for (int run = 0; run < NUM_RUNS; ++run) {
         cudaEventRecord(start);

         peak_flops_fp64<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);

         cudaEventRecord(stop);
         cudaEventSynchronize(stop);

         float milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);

         long long total_threads = (long long)THREADS_PER_BLOCK * NUM_BLOCKS;
         long long total_flops = total_threads * FLOP_ITERATIONS * 4; // 2 MULs and 2 ADDs per iteration

         double giga_flops = (double)total_flops / (milliseconds * 1e6);
         gflops_results[run] = giga_flops;
      }

      double mean, stddev;
      calculate_statistics(gflops_results, NUM_RUNS, &mean, &stddev);

      printf("FP64 Performance over %d runs:  %.2f  +/- %.2f\n", NUM_RUNS, mean, stddev);

      cudaFree(d_out);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
   }

   // Measure FP32 Performance
   {
      // printf("Measuring FP32 peak FLOPS...\n");

      float *d_out;
      cudaMalloc((void **)&d_out, THREADS_PER_BLOCK * NUM_BLOCKS * sizeof(float));

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      double gflops_results[NUM_RUNS];

      // Warm-up run
      peak_flops_fp32<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);
      cudaDeviceSynchronize();

      for (int run = 0; run < NUM_RUNS; ++run) {
         cudaEventRecord(start);

         peak_flops_fp32<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);

         cudaEventRecord(stop);
         cudaEventSynchronize(stop);

         float milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);

         long long total_threads = (long long)THREADS_PER_BLOCK * NUM_BLOCKS;
         long long total_flops = total_threads * FLOP_ITERATIONS * 4; // 2 MULs and 2 ADDs per iteration

         double giga_flops = (double)total_flops / (milliseconds * 1e6);
         gflops_results[run] = giga_flops;
      }

      double mean, stddev;
      calculate_statistics(gflops_results, NUM_RUNS, &mean, &stddev);

      printf("FP32 Performance over %d runs:  %.2f  +/- %.2f\n", NUM_RUNS, mean, stddev);

      cudaFree(d_out);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
   }

   // Measure FP16 Performance
   {
      // printf("Measuring FP16 peak FLOPS...\n");

      __half *d_out;
      cudaMalloc((void **)&d_out, THREADS_PER_BLOCK * NUM_BLOCKS * sizeof(__half));

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      double gflops_results[NUM_RUNS];

      // Warm-up run
      peak_flops_fp16<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);
      cudaDeviceSynchronize();

      for (int run = 0; run < NUM_RUNS; ++run) {
         cudaEventRecord(start);

         peak_flops_fp16<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);

         cudaEventRecord(stop);
         cudaEventSynchronize(stop);

         float milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);

         long long total_threads = (long long)THREADS_PER_BLOCK * NUM_BLOCKS;
         long long total_flops = total_threads * FLOP_ITERATIONS * 4; // 2 MULs and 2 ADDs per iteration

         double giga_flops = (double)total_flops / (milliseconds * 1e6);
         gflops_results[run] = giga_flops;
      }

      double mean, stddev;
      calculate_statistics(gflops_results, NUM_RUNS, &mean, &stddev);

      printf("FP16 Performance over %d runs:  %.2f  +/- %.2f\n", NUM_RUNS, mean, stddev);

      cudaFree(d_out);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
   }

#ifdef FP8
   // Measure FP8 Performance
   {
      // printf("Measuring FP8 peak FLOPS...\n");

      __nv_fp8_e4m3 *d_out;
      cudaMalloc((void **)&d_out, THREADS_PER_BLOCK * NUM_BLOCKS * sizeof(__nv_fp8_e4m3));

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      double gflops_results[NUM_RUNS];

      // Warm-up run
      peak_flops_fp8<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);
      cudaDeviceSynchronize();

      for (int run = 0; run < NUM_RUNS; ++run) {
         cudaEventRecord(start);

         peak_flops_fp8<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);

         cudaEventRecord(stop);
         cudaEventSynchronize(stop);

         float milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);

         long long total_threads = (long long)THREADS_PER_BLOCK * NUM_BLOCKS;
         long long total_flops = total_threads * FLOP_ITERATIONS * 4; // Adjust if needed

         double giga_flops = (double)total_flops / (milliseconds * 1e6);
         gflops_results[run] = giga_flops;
      }

      double mean, stddev;
      calculate_statistics(gflops_results, NUM_RUNS, &mean, &stddev);

      printf("FP8 Performance over %d runs:  %.2f  +/- %.2f\n", NUM_RUNS, mean, stddev);

      cudaFree(d_out);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
   }
#endif

   return 0;
}

// GPU Device:         NVIDIA H100 80GB HBM3
// Compute Capability: 9.0
// FLOP_ITERATIONS:     1000000
// THREADS_PER_BLOCK:   512
// NUM_BLOCKS:          10000
// NUM_RUNS:            10
// FP64 Performance over 10 runs:  11097.92  +/- 0.45
// FP32 Performance over 10 runs:  21166.94  +/- 0.87
// FP16 Performance over 10 runs:  15843.49  +/- 0.72

// GPU Device:         NVIDIA GH200 480GB
// Compute Capability: 9.0
// FLOP_ITERATIONS:     1000000
// THREADS_PER_BLOCK:   512
// NUM_BLOCKS:          10000
// NUM_RUNS:            10
// FP64 Performance over 10 runs:  11101.06  +/- 0.01
// FP32 Performance over 10 runs:  21170.34  +/- 0.01
// FP16 Performance over 10 runs:  15848.30  +/- 0.18

// GPU Device:         NVIDIA A100-PCIE-40GB
// Compute Capability: 8.0
// FLOP_ITERATIONS:     1000000
// THREADS_PER_BLOCK:   256
// NUM_BLOCKS:          10000
// NUM_RUNS:            10
// FP64 Performance over 10 runs:  4846.84  +/- 0.01
// FP32 Performance over 10 runs:  9684.21  +/- 0.01
// FP16 Performance over 10 runs:  9207.34  +/- 0.82

// GPU Device:         Tesla V100-SXM2-32GB
// Compute Capability: 7.0
// FLOP_ITERATIONS:     1000000
// THREADS_PER_BLOCK:   256
// NUM_BLOCKS:          4000
// NUM_RUNS:            10
// FP64 Performance over 10 runs:  3644.34  +/- 30.17
// FP32 Performance over 10 runs:  7209.37  +/- 64.88
// FP16 Performance over 10 runs:  5998.10  +/- 0.14

// GPU Device:         Tesla P100-PCIE-16GB
// Compute Capability: 6.0
// FLOP_ITERATIONS:     1000000
// THREADS_PER_BLOCK:   256
// NUM_BLOCKS:          4000
// NUM_RUNS:            10
// FP64 Performance over 10 runs:  2326.07  +/- 9.38
// FP32 Performance over 10 runs:  4653.07  +/- 31.81
// FP16 Performance over 10 runs:  4547.42  +/- 0.46