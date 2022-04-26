/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * April 2022
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <functional>

#include "torch/extension.h"
#include <cuda.h>
#include "diceloss.h"

typedef c10::IntArrayRef IntArrayRef;

// From: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// And from: https://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

//#if __CUDA_ARCH__ < 600
#else
static inline __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

namespace {

template<typename RealType>
__device__ void softmax(RealType *d_outData, const RealType *d_inData, int64_t i64NumChannels, int64_t i64Stride) {
  int64_t cmax = 0;
  for (int64_t c = 1; c < i64NumChannels; ++c) {
    if (d_inData[i64Stride*cmax] < d_inData[i64Stride*c])
      cmax = c;
  }

  const RealType offset = d_inData[i64Stride*cmax];

  RealType norm = RealType(0);
  for (int64_t c = 0; c < i64NumChannels; ++c) {
    d_outData[c] = exp(d_inData[i64Stride*c] - offset);
    norm += d_outData[c];
  }

  for (int64_t c = 0; c < i64NumChannels; ++c)
    d_outData[c] /= norm;
}

__global__ void CheckForInvalidLabels(const int64_t *d_i64InMask, volatile bool *d_bInvalidFound, int64_t i64IgnoreLabel, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label != i64IgnoreLabel && (i64Label < 0 || i64Label >= i64NumChannels))
      *d_bInvalidFound = true;
  }
}

template<typename RealType>
__global__ void DiceLossForwardKernel(const RealType *d_inData, const int64_t *d_i64InMask, RealType *d_num, RealType *d_den, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, RealType smooth, int p, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  // Workaround shared memory limitation in template functions... as described here:
  // https://stackoverflow.com/questions/20497209/getting-cuda-error-declaration-is-incompatible-with-previous-variable-name
  extern __shared__ uint8_t s[]; // Yes, would have been nice to have it as RealType s[]...

  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  // Work memory for softmax()
  RealType * const d_values = ((RealType *)s) + i64NumChannels*(threadIdx.y * blockDim.x + threadIdx.x);

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label == i64IgnoreLabel)
      return;

    softmax(d_values, d_inData + (b*i64NumChannels + 0)*i64InnerDataNum + i, i64NumChannels, i64InnerDataNum);

    for (int64_t c = 0; c < i64NumChannels; ++c) {
      if (c != i64IgnoreChannel) {
        const RealType binaryLabel = RealType(c == i64Label ? 1 : 0);
        // XXX: This sucks!
        atomicAdd(d_num + (b*i64NumChannels + c), d_values[c]*binaryLabel);
        atomicAdd(d_den + (b*i64NumChannels + c), pow(d_values[c], p) + binaryLabel);
      }
    }
  }
}

template<typename RealType>
__global__ void DiceLossBackwardKernelData(const RealType *d_inData, const int64_t *d_i64InMask, const RealType *d_num, const RealType *d_den, const RealType *d_outLossGrad, RealType *d_inDataGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, RealType smooth, int p, ReductionType eReduction, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  // Workaround shared memory limitation in template functions... as described here:
  // https://stackoverflow.com/questions/20497209/getting-cuda-error-declaration-is-incompatible-with-previous-variable-name
  extern __shared__ uint8_t s[]; // Yes, would have been nice to have it as RealType s[]...

  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  // Work memory for softmax()
  RealType * const d_values = ((RealType *)s) + i64NumChannels*(threadIdx.y * blockDim.x + threadIdx.x);

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label == i64IgnoreLabel)
      return;

    RealType scale = RealType(1);

    switch (eReduction) {
    case NoneReduction:
      scale = d_outLossGrad[b] / RealType(i64NumChannels);
      break;
    case MeanReduction:
      scale = *d_outLossGrad / RealType(i64NumChannels * i64BatchSize);
      break;
    case SumReduction:
      scale = *d_outLossGrad / RealType(i64NumChannels);
      break;
    default:
      break;
    }

    softmax(d_values, d_inData + (b*i64NumChannels + 0)*i64InnerDataNum + i, i64NumChannels, i64InnerDataNum);

    for (int64_t c = 0; c < i64NumChannels; ++c) {
      if (c == i64IgnoreChannel)
        continue;

      for (int64_t c2 = 0; c2 < i64NumChannels; ++c2) {
        if (c2 == i64IgnoreChannel)
          continue;

        const RealType num = d_num[b*i64NumChannels + c2];
        const RealType den = d_den[b*i64NumChannels + c2];

        const RealType delta = RealType(c == c2 ? 1 : 0);
        const RealType binaryLabel = RealType(i64Label == c2 ? 1 : 0);

        switch (p) {
        case 1:
          d_inDataGrad[(b*i64NumChannels + c)*i64InnerDataNum + i] -= (RealType(2)*den*binaryLabel - num)/(den*den) * (delta - d_values[c2]); // NOTE: Missing d_values[c] factor is moved below
          break;
        default:
          d_inDataGrad[(b*i64NumChannels + c)*i64InnerDataNum + i] -= (RealType(2)*den*binaryLabel - RealType(p)*num*pow(d_values[c2],p-1))/(den*den) * (delta - d_values[c2]);
          break;
        }
      }

      d_inDataGrad[(b*i64NumChannels + c)*i64InnerDataNum + i] *= scale * d_values[c];
    }
  }
}

} // end anonymous namespace

template<typename RealType>
torch::Tensor diceloss_gpu_forward(torch::Tensor inData, torch::Tensor inMask, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction) {
  // NOTE: kInt64 should match torch.long
  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return torch::Tensor();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return torch::Tensor();


  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  int64_t i64InnerDataNum = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), IntArrayRef::value_type(1), std::multiplies<IntArrayRef::value_type>());
  }
  
  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  const dim3 threadsPerBlock(64,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y - 1)/threadsPerBlock.y);
  const size_t sharedMem = i64NumChannels*threadsPerBlock.x*threadsPerBlock.y*sizeof(RealType);

  const RealType * const d_inData = inData.data_ptr<RealType>();
  const int64_t * const d_i64InMask = inMask.data_ptr<int64_t>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, i64NumChannels, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return torch::Tensor();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);

  RealType * const d_num = num.data_ptr<RealType>();
  RealType * const d_den = den.data_ptr<RealType>();

  DiceLossForwardKernel<RealType><<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, smooth, p, i64BatchSize, i64NumChannels, i64InnerDataNum);

  torch::Tensor outLoss = RealType(1) - (RealType(2)*num + smooth)/(den + smooth);
  outLoss = outLoss.mean(IntArrayRef(1));

  switch (eReduction) {
  case NoneReduction:
    return outLoss;
  case MeanReduction:
    return outLoss.mean();
  case SumReduction:
    return outLoss.sum();
  default:
    return torch::Tensor(); // Uhh?
  }

  return torch::Tensor(); // Not reached
}

template<typename RealType>
std::vector<torch::Tensor> diceloss_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction) {
  if (bInMaskGrad) // This is never differentiable!
    return std::vector<torch::Tensor>(); 

  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return std::vector<torch::Tensor>();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return std::vector<torch::Tensor>();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  int64_t i64InnerDataNum = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), IntArrayRef::value_type(1), std::multiplies<IntArrayRef::value_type>());
  }

  switch (eReduction) {
  case NoneReduction:
    if (outLossGrad.sizes() != IntArrayRef(i64BatchSize))
      return std::vector<torch::Tensor>();
    break;
  case MeanReduction:
  case SumReduction: // Fall through
    if (outLossGrad.numel() != 1)
      return std::vector<torch::Tensor>();
    break;
  default: // Uhh?
    return std::vector<torch::Tensor>();
  }

  std::vector<torch::Tensor> vGradTensors(2);

  // Avoid doing any computation!
  if (!bInDataGrad && !bInMaskGrad) // The mask grad is always false... but it's there for clarity!
    return vGradTensors;

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  const dim3 threadsPerBlock(64,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y - 1)/threadsPerBlock.y);
  const size_t sharedMem = i64NumChannels*threadsPerBlock.x*threadsPerBlock.y*sizeof(RealType);

  const RealType * const d_inData = inData.data_ptr<RealType>();
  const int64_t * const d_i64InMask = inMask.data_ptr<int64_t>();
  const RealType * const d_outLossGrad = outLossGrad.data_ptr<RealType>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, i64NumChannels, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return std::vector<torch::Tensor>();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);

  RealType * const d_num = num.data_ptr<RealType>();
  RealType * const d_den = den.data_ptr<RealType>();

  // Calculate d_num and d_den... we can't ignore any channel here!
  DiceLossForwardKernel<RealType><<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, smooth, p, i64BatchSize, i64NumChannels, i64InnerDataNum);

  num *= RealType(2);
  num += smooth;
  den += smooth;

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const d_inDataGrad = inDataGrad.data_ptr<RealType>();

    DiceLossBackwardKernelData<RealType><<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_num, d_den, d_outLossGrad, d_inDataGrad, i64IgnoreChannel, i64IgnoreLabel, smooth, p, eReduction, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[0] = inDataGrad;
  }

  return vGradTensors;
}

template torch::Tensor diceloss_gpu_forward<float>(torch::Tensor, torch::Tensor, int64_t, int64_t i64IgnoreLabel, const float &, int, ReductionType);
template torch::Tensor diceloss_gpu_forward<double>(torch::Tensor, torch::Tensor, int64_t, int64_t i64IgnoreLabel, const double &, int, ReductionType);

template std::vector<torch::Tensor> diceloss_gpu_backward<float>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, int64_t, int64_t, const float &, int, ReductionType);
template std::vector<torch::Tensor> diceloss_gpu_backward<double>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, int64_t, int64_t, const double &, int, ReductionType);

