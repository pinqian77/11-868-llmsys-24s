#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32

template <typename T>
__forceinline__ __device__ T add_eps(T x) {
  return fabsf(x) > LN_EPSILON ? x : (x < 0 ? -LN_EPSILON : LN_EPSILON);
}

/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  // if (isnan(inp[0]) || isnan(inp[1])) {
  //   printf("NaN Detected! \n");
  // }

  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;

  // if (isnan(inp_f4[0].x) || isnan(inp_f4[0].y) || isnan(inp_f4[0].z) || isnan(inp_f4[0].w)) {
  //   printf("NaN Detected! inp_f4 \n");
  //   printf("inp: %f, %f, %f, %f\n", inp[0], inp[1], inp[2], inp[3]);
  // }

  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];

    l_sum += val.x + val.y + val.z + val.w;
    l_square_sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;

    // if (isnan(l_sum) || isnan(l_square_sum)) {
    //   printf("val.x: %f, val.y: %f, val.z: %f, val.w: %f\n", val.x, val.y, val.z, val.w);
    // }
  }

  // Step 2
  float mean_dim = float(hidden_size) * 4.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);

  // if (isnan(reduce_val[0]) || isnan(reduce_val[1])) {
  //   printf("------------------------------------------------------- \n");
  //   printf("val.x: %f, val.y: %f, val.z: %f, val.w: %f\n", val.x, val.y, val.z, val.w);
  //   printf("l_sum: %f, l_square_sum: %f\n", l_sum, l_square_sum);
  //   printf("reduce_val[0]: %f, reduce_val[1]: %f\n", reduce_val[0], reduce_val[1]);
  //   printf("mean_dim: %f\n", mean_dim);
  //   // Optionally, print more diagnostics
  // }

  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + LN_EPSILON;
    // if (isnan(s_mean) || isnan(s_var)) {
    //     printf("NaN Detected! \n");
    //     printf("Block: %d, Thread: %d\n", blockIdx.x, threadIdx.x);
    //     printf("l_sum: %f, l_square_sum: %f\n", l_sum, l_square_sum);
    //     printf("reduce_val[0]: %f, reduce_val[1]: %f\n", reduce_val[0], reduce_val[1]);
    //     printf("mean_dim: %f\n", mean_dim);
    //     printf("s_mean: %f, s_var: %f\n", s_mean, s_var);
    //     // Optionally, print more diagnostics
    // }
    vars[blockIdx.x] = s_var;
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  // Step 3
  float4 *output_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 vscale = __ldg(reinterpret_cast<const float4 *>(scale) + idx);
    float4 vbias = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
    float4 val = inp_f4[idx];
    val.x = (val.x - s_mean) * s_var * vscale.x + vbias.x;
    val.y = (val.y - s_mean) * s_var * vscale.y + vbias.y;
    val.z = (val.z - s_mean) * s_var * vscale.z + vbias.z;
    val.w = (val.w - s_mean) * s_var * vscale.w + vbias.w;
    output_f4[idx] = val;
  }
  
  /// END ASSIGN3_2
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = threadIdx.y * width + idx;
  int y_stride = width * TILE_DIM;

  // Step 1
  float dbetta = 0;
  float dgamma = 0;
  float dout, val;

  float thread_cmax_g = 0;
  if (idx < width) {
    if (means == nullptr) {
      float vbetta = (float)betta[idx];
      float vgamma = (float)gamma[idx];
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = (float)out_grad[offset];

        val = (float)inp[offset];
        dbetta += dout;
        dgamma += ((val - vbetta) / add_eps(vgamma) * dout);
        offset += y_stride;
      }
    } 
    else {
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = (float)out_grad[offset];
        val = (float)inp[offset];
        dbetta += dout;
        dgamma += ((val - (float)means[r]) * rsqrtf((float)vars[r] + LN_EPSILON) * dout);
        offset += y_stride;
      }
    }
  }

  // Step 2
  __shared__ float block_cmax_g;
  if (threadIdx.x == 0 && threadIdx.y == 0) block_cmax_g = 0;

  betta_buffer[threadIdx.x][threadIdx.y] = dbetta;
  gamma_buffer[threadIdx.x][threadIdx.y] = dgamma;
  __syncthreads();

  if (thread_cmax_g != 0) {
    atomicAdd(&block_cmax_g, thread_cmax_g);
  }

  float s1 = betta_buffer[threadIdx.y][threadIdx.x];
  float s2 = gamma_buffer[threadIdx.y][threadIdx.x];
  __syncthreads();

  // Step 3
  for (int i = 1; i < TILE_DIM; i <<= 1) {
    s1 += g.shfl_down(s1, i);
    s2 += g.shfl_down(s2, i);
  }

  // Step 4
  int pos = blockIdx.x * TILE_DIM + threadIdx.y;
  if (threadIdx.x == 0 && idx < width) {
    betta_grad[pos] = s1;
    gamma_grad[pos] = s2;
  }
  /// END ASSIGN3_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient

  // step 1
  int offset = blockIdx.x * hidden_dim + threadIdx.x;
  float4 dxhat, xhat;
  float var_rsqrt;
  if (threadIdx.x < hidden_dim) {
    dxhat = ((const float4 *)out_grad)[offset];
    float4 vgamma = ((const float4 *)gamma)[threadIdx.x];
    dxhat.x *= vgamma.x;
    dxhat.y *= vgamma.y;
    dxhat.z *= vgamma.z;
    dxhat.w *= vgamma.w;

    // step 2
    xhat = ((const float4 *)inp)[offset];
    var_rsqrt = rsqrtf((float)vars[blockIdx.x] + LN_EPSILON);
    if (means == nullptr) {
      float4 vbetta = ((const float4 *)betta)[threadIdx.x];
      xhat.x = (xhat.x - vbetta.x) / add_eps(vgamma.x);
      xhat.y = (xhat.y - vbetta.y) / add_eps(vgamma.y);
      xhat.z = (xhat.z - vbetta.z) / add_eps(vgamma.z);
      xhat.w = (xhat.w - vbetta.w) / add_eps(vgamma.w);
    } 
    else {
      float fmean = (float)means[blockIdx.x];
      xhat.x = (xhat.x - fmean) * var_rsqrt;
      xhat.y = (xhat.y - fmean) * var_rsqrt;
      xhat.z = (xhat.z - fmean) * var_rsqrt;
      xhat.w = (xhat.w - fmean) * var_rsqrt;
    }
  }

  // step 3
  float reduce_val[2] = {0.f, 0.f};
  if (threadIdx.x < hidden_dim) {
    reduce_val[0] = dxhat.x + dxhat.y + dxhat.z + dxhat.w;
    reduce_val[1] = dxhat.x * xhat.x + dxhat.y * xhat.y + dxhat.z * xhat.z +
                    dxhat.w * xhat.w;
  }
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_sum_dxhat, s_sum_dxhat_xhat;
  if (threadIdx.x == 0) {
    float mean_dim = hidden_dim * 4;
    s_sum_dxhat = reduce_val[0] / mean_dim;
    s_sum_dxhat_xhat = reduce_val[1] / mean_dim;
  }
  __syncthreads();

  // step 4
  if (threadIdx.x >= hidden_dim) {
    return;
  }
  dxhat.x = (dxhat.x - s_sum_dxhat - xhat.x * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.y = (dxhat.y - s_sum_dxhat - xhat.y * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.z = (dxhat.z - s_sum_dxhat - xhat.z * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.w = (dxhat.w - s_sum_dxhat - xhat.w * s_sum_dxhat_xhat) * var_rsqrt;

  ((float4 *)inp_grad)[offset] = dxhat;
  
  /// END ASSIGN3_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
