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

#include "cuda_fp16.h"

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
#endif // atomicAdd

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)

#else
static inline __device__ __half2 __hmax2(__half2 a, __half2 b) {
  const __half2 res = __hgt2(a, b);
  const __half max1 = __half_as_ushort(__low2half(res)) ? __low2half(a) : __low2half(b);
  const __half max2 = __half_as_ushort(__high2half(res)) ? __high2half(a) : __high2half(b);
  return __halves2half2(max1, max2);
}
#endif // half2 intrinsics

namespace {

__device__ __half2 myh2pow(__half2 x, int p) {
  switch (p) {
  case 1:
    return x;
  case 2:
    return __hmul2(x, x);
  default:
    return h2exp(__hmul2(__float2half2_rn((float)p), h2log(x)));
  }
  return __half2(); // Not reached
}

__device__ __half myhpow(__half x, int p) {
  switch (p) {
  case 1:
    return x;
  case 2:
    return __hmul(x, x);
  default:
    return hexp(__hmul(__float2half((float)p), hlog(x)));
  }
  return __half(); // Not reached
}

template<typename RealType>
__device__ void softmax(RealType *d_outData, const RealType *d_inData, int64_t i64NumChannels, int64_t i64Stride) {
  RealType offset = d_inData[i64Stride*0];
  for (int64_t c = 1; c < i64NumChannels; ++c) {
    if (offset < d_inData[i64Stride*c])
      offset = d_inData[i64Stride*c];
  }

  RealType norm = RealType(0);
  for (int64_t c = 0; c < i64NumChannels; ++c) {
    d_outData[c] = exp(d_inData[i64Stride*c] - offset);
    norm += d_outData[c];
  }

  for (int64_t c = 0; c < i64NumChannels; ++c)
    d_outData[c] /= norm;
}

template<>
__device__ void softmax<__half>(__half *d_outData, const __half *d_inData, int64_t i64NumChannels, int64_t i64Stride) {
  const int64_t i64NumChannels2 = i64NumChannels / 2;

  const __half zero = __ushort_as_half(0);
  const __half2 zero2 = __half2half2(zero);

  __half2 offset2 = __half2half2(d_inData[(i64NumChannels-1)*i64Stride]);

  for (int64_t c2 = 0; c2 < i64NumChannels2; ++c2) {
    const half2 tmp2 = __halves2half2(d_inData[(2*c2+0)*i64Stride], d_inData[(2*c2+1)*i64Stride]);
    offset2 = __hmax2(offset2, tmp2);
  }

  offset2 = __hmax2(offset2, __lowhigh2highlow(offset2));

  __half2 norm2 = zero2;

  for (int64_t c2 = 0; c2 < i64NumChannels2; ++c2) {
    __half2 tmp2 = __halves2half2(d_inData[(2*c2+0)*i64Stride], d_inData[(2*c2+1)*i64Stride]);
    tmp2 = h2exp(__hsub2(tmp2, offset2));
    norm2 = __hadd2(norm2, tmp2);
    d_outData[2*c2 + 0] = __low2half(tmp2); 
    d_outData[2*c2 + 1] = __high2half(tmp2); 
  }

  if (2*i64NumChannels2 != i64NumChannels) {
    __half tmp = d_inData[(i64NumChannels-1)*i64Stride];
    tmp = hexp(__hsub(tmp, __low2half(offset2)));
    norm2 = __hadd2(norm2, __halves2half2(tmp, zero));
    d_outData[i64NumChannels-1] = tmp;
  }

  norm2 = __hadd2(norm2, __lowhigh2highlow(norm2));

  for (int64_t c2 = 0; c2 < i64NumChannels2; ++c2) {
    __half2 tmp2 = __h2div(__halves2half2(d_outData[2*c2 + 0], d_outData[2*c2 + 1]), norm2);
    d_outData[2*c2 + 0] = __low2half(tmp2);
    d_outData[2*c2 + 1] = __high2half(tmp2);
  }

  if (2*i64NumChannels2 != i64NumChannels)
    d_outData[i64NumChannels-1] = __hdiv(d_outData[i64NumChannels-1], __low2half(norm2));
}

template<typename RealType>
__device__ RealType sigmoid(const RealType &x) {
  if (x < RealType(0)) {
    const RealType tmp = exp(x);
    return tmp/(RealType(1) + tmp);
  }

  return RealType(1)/(RealType(1) + exp(-x));
}

template<>
__device__ __half sigmoid<__half>(const __half &x) {
  const __half zero = __ushort_as_half(0);
  const __half one = __ushort2half_rn(1);

  if (__hlt(x, zero)) {
    const __half tmp = hexp(x);
    return __hdiv(tmp, __hadd(one, tmp));
  }

  //return __hdiv(one, __hadd(one, hexp(__hneg(x))));
  return hrcp(__hadd(one, hexp(__hneg(x))));
}

template<typename RealType>
__device__ RealType dsigmoid(const RealType &x, RealType &s) {
  // s'(x) = s(x)**2 * exp(-x)
  //       = (1/(exp(x/2) + exp(-x/2)))**2
  //       = s(-x)**2 * exp(x)
  if (x < RealType(0)) {
    const RealType tmp = exp(x);
    s = tmp/(RealType(1) + tmp);
    return tmp/pow(RealType(1) + tmp, 2);
  }

  const RealType tmp = exp(-x);
  s = RealType(1)/(RealType(1) + tmp);
  return tmp/pow(RealType(1) + tmp, 2);
}

template<>
__device__ __half dsigmoid<__half>(const __half &x, __half &s) {
  const __half zero = __ushort_as_half(0);
  const __half one = __ushort2half_rn(1);

  if (__hlt(x, zero)) {
    const __half tmp = hexp(x);
    s = __hdiv(tmp, __hadd(one, tmp));
    return __hdiv(tmp, myhpow(__hadd(one, tmp), 2));
  }

  const __half tmp = hexp(__hneg(x));
  //s = __hdiv(one, __hadd(one, tmp));
  s = hrcp(__hadd(one, tmp));
  return __hdiv(tmp, myhpow(__hadd(one, tmp), 2));
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
__global__ void DiceLossForwardKernel(const RealType *d_inData, const int64_t *d_i64InMask, RealType *d_num, RealType *d_den, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, int p, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
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
__global__ void BinaryDiceLossForwardKernel(const RealType *d_inData, const int64_t *d_i64InMask, RealType *d_num, RealType *d_den, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, int p, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label == i64IgnoreLabel)
      return;

    const RealType s = sigmoid(d_inData[(b*i64NumChannels + 0)*i64InnerDataNum + i]);
    const RealType a_values[2] = { RealType(1) - s, s };

    for (int64_t c = 0; c < 2; ++c) {
      if (c != i64IgnoreChannel) {
        const RealType binaryLabel = RealType(c == i64Label ? 1 : 0);
        // XXX: This sucks!
        atomicAdd(d_num + (b*2 + c), a_values[c]*binaryLabel);
        atomicAdd(d_den + (b*2 + c), pow(a_values[c], p) + binaryLabel);
      }
    }
  }
}

__global__ void DiceLossForwardKernelHalf(const __half *d_inData, const int64_t *d_i64InMask, float *d_num, float *d_den, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, int p, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  extern __shared__ uint8_t s[]; 

  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  // Work memory for softmax()
  __half * const d_values = ((__half *)s) + i64NumChannels*(threadIdx.y * blockDim.x + threadIdx.x);

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label == i64IgnoreLabel)
      return;

    softmax(d_values, d_inData + (b*i64NumChannels + 0)*i64InnerDataNum + i, i64NumChannels, i64InnerDataNum);

    const int64_t i64NumChannels2 = i64NumChannels / 2;
    const __half zero = __ushort_as_half(0);

    for (int64_t c2 = 0; c2 < i64NumChannels2; ++c2) {
      __half2 values2;
      __half2 binaryLabel2;

      switch (i64IgnoreChannel - 2*c2) {
      case 0:
        values2 = __halves2half2(zero, d_values[2*c2 + 1]);
        binaryLabel2 = __halves2half2(zero,  __ushort2half_rn(2*c2 + 1 == i64Label ? 1 : 0));
        break;
      case 1:
        values2 = __halves2half2(d_values[2*c2 + 0], zero);
        binaryLabel2 = __halves2half2(__ushort2half_rn(2*c2 + 0 == i64Label ? 1 : 0), zero);
        break;
      default:
        values2 = __halves2half2(d_values[2*c2 + 0], d_values[2*c2 + 1]);
        binaryLabel2 = __halves2half2(
            __ushort2half_rn(2*c2 + 0 == i64Label ? 1 : 0),
            __ushort2half_rn(2*c2 + 1 == i64Label ? 1 : 0)
        );
        break;
      }

      const __half2 numTerms = __hmul2(values2, binaryLabel2);
      const __half2 denTerms = __hadd2(myh2pow(values2, p), binaryLabel2);

      atomicAdd(d_num + (b*i64NumChannels + (2*c2 + 0)), __low2float(numTerms));
      atomicAdd(d_num + (b*i64NumChannels + (2*c2 + 1)), __high2float(numTerms));

      atomicAdd(d_den + (b*i64NumChannels + (2*c2 + 0)), __low2float(denTerms));
      atomicAdd(d_den + (b*i64NumChannels + (2*c2 + 1)), __high2float(denTerms));
    }

    if (2*i64NumChannels2 != i64NumChannels && i64NumChannels-1 != i64IgnoreChannel) {
      const __half binaryLabel = __ushort2half_rn(i64NumChannels-1 == i64Label ? 1 : 0);

      const __half numTerm = __hmul(d_values[i64NumChannels-1], binaryLabel);
      const __half denTerm = __hadd(myhpow(d_values[i64NumChannels-1], p), binaryLabel);

      atomicAdd(d_num + (b*i64NumChannels + (i64NumChannels-1)), __half2float(numTerm));
      atomicAdd(d_den + (b*i64NumChannels + (i64NumChannels-1)), __half2float(denTerm));
    }
  }
}

__global__ void BinaryDiceLossForwardKernelHalf(const __half *d_inData, const int64_t *d_i64InMask, float *d_num, float *d_den, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, int p, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label == i64IgnoreLabel)
      return;

    const __half s = sigmoid(d_inData[(b*i64NumChannels + 0)*i64InnerDataNum + i]);

    const __half zero = __ushort_as_half(0);
    const __half one = __ushort2half_rn(1);

    __half2 values2;
    __half2 binaryLabel2;

    switch (i64IgnoreChannel) {
    case 0:
      values2 = __halves2half2(zero, s);
      binaryLabel2 = __halves2half2(zero,  __ushort2half_rn(1 == i64Label ? 1 : 0));
      break;
    case 1:
      values2 = __halves2half2(__hsub(one,s), zero);
      binaryLabel2 = __halves2half2(__ushort2half_rn(0 == i64Label ? 1 : 0), zero);
      break;
    default:
      values2 = __halves2half2(__hsub(one,s), s);
      binaryLabel2 = __halves2half2(
          __ushort2half_rn(0 == i64Label ? 1 : 0),
          __ushort2half_rn(1 == i64Label ? 1 : 0)
      );
      break;
    }

    const __half2 numTerms = __hmul2(values2, binaryLabel2);
    const __half2 denTerms = __hadd2(myh2pow(values2, p), binaryLabel2);

    atomicAdd(d_num + (b*2 +  0), __low2float(numTerms));
    atomicAdd(d_num + (b*2 +  1), __high2float(numTerms));

    atomicAdd(d_den + (b*2 + 0), __low2float(denTerms));
    atomicAdd(d_den + (b*2 + 1), __high2float(denTerms));
  }
}

// Kernel to deal with corner case
template<typename RealType>
__global__ void DiceLossForwardKernel2(const RealType *d_num, const RealType *d_den, RealType *d_outLoss, int64_t i64BatchSize, int64_t i64NumChannels) {
  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (b < i64BatchSize && c < i64NumChannels) {
    const RealType num = d_num[b*i64NumChannels + c];
    const RealType den = d_den[b*i64NumChannels + c];

    // By definition: den >= num and we say DSC = 1 when comparing two empty sets (which are the same)
    if (den == RealType(0))
      d_outLoss[b*i64NumChannels + c] = RealType(0);
    else
      d_outLoss[b*i64NumChannels + c] = RealType(1) - num/den;
  }
}

template<typename RealType>
__global__ void DiceLossBackwardKernelData(const RealType *d_inData, const int64_t *d_i64InMask, const RealType *d_inWeight, const RealType *d_num, const RealType *d_den, const RealType *d_outLossGrad, RealType *d_inDataGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, int p, ReductionType eReduction, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
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
      scale = d_outLossGrad[b];
      break;
    case MeanReduction:
      scale = *d_outLossGrad / RealType(i64BatchSize);
      break;
    case SumReduction:
    case BatchDiceReduction: // Fall through
      scale = *d_outLossGrad;
      break;
    default:
      break;
    }

    if (eReduction != BatchDiceReduction) {
      d_num += b*i64NumChannels;
      d_den += b*i64NumChannels;
    }

    softmax(d_values, d_inData + (b*i64NumChannels + 0)*i64InnerDataNum + i, i64NumChannels, i64InnerDataNum);

    for (int64_t c = 0; c < i64NumChannels; ++c) {
      const RealType weight = d_inWeight != nullptr ? d_inWeight[c] : RealType(1)/RealType(i64NumChannels);

      // NOTE:  Owing to the Jacobian multiplication, we can't ignore c == ignore_channel here!
      RealType tmp = RealType(0);
      for (int64_t c2 = 0; c2 < i64NumChannels; ++c2) {
        if (c2 == i64IgnoreChannel) // This loss term is discarded (constant), so the partial = 0 for this term
          continue;

        const RealType num = d_num[c2];
        const RealType den = d_den[c2];

        const RealType delta = RealType(c == c2 ? 1 : 0);
        const RealType binaryLabel = RealType(i64Label == c2 ? 1 : 0);

        // Recall den >= num
        if (den == RealType(0)) // If num == 0, the DSC curve is a constant 0, so we say the limiting case has 0 derivative
          continue;

        switch (p) {
        case 1:
          tmp -= (RealType(2)*den*binaryLabel - num)/(den*den) * (delta - d_values[c2]); // NOTE: Missing d_values[c] factor is moved below
          break;
        default:
          tmp -= (RealType(2)*den*binaryLabel - RealType(p)*num*pow(d_values[c2],p-1))/(den*den) * (delta - d_values[c2]);
          break;
        }
      }

      tmp *= scale * weight * d_values[c];
      d_inDataGrad[(b*i64NumChannels + c)*i64InnerDataNum + i] = tmp;
    }
  }
}

template<typename RealType>
__global__ void BinaryDiceLossBackwardKernelData(const RealType *d_inData, const int64_t *d_i64InMask, const RealType *d_inWeight, const RealType *d_num, const RealType *d_den, const RealType *d_outLossGrad, RealType *d_inDataGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, int p, ReductionType eReduction, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label == i64IgnoreLabel)
      return;

    RealType scale = RealType(1);

    switch (eReduction) {
    case NoneReduction:
      scale = d_outLossGrad[b];
      break;
    case MeanReduction:
      scale = *d_outLossGrad / RealType(i64BatchSize);
      break;
    case SumReduction:
    case BatchDiceReduction: // Fall through
      scale = *d_outLossGrad;
      break;
    default:
      break;
    }

    if (eReduction != BatchDiceReduction) {
      d_num += b*2;
      d_den += b*2;
    }

    RealType s = RealType(0);
    const RealType ds = dsigmoid(d_inData[(b*i64NumChannels + 0)*i64InnerDataNum + i], s);
    const RealType a_values[2] = { RealType(1) - s, s };
    const RealType a_dvalues[2] = { -ds, ds };

    RealType tmp = RealType(0);

    for (int64_t c = 0; c < 2; ++c) {
      if (c == i64IgnoreChannel)
        continue;

      const RealType weight = d_inWeight != nullptr ? d_inWeight[c] : RealType(0.5);

      const RealType num = d_num[c];
      const RealType den = d_den[c];
      const RealType binaryLabel = RealType(i64Label == c ? 1 : 0);

      // Recall den >= num
      if (den == RealType(0)) // If num == 0, the DSC curve is a constant 0, so we say the limiting case has 0 derivative
        continue;

      switch (p) {
      case 1:
        tmp -= (RealType(2)*den*binaryLabel - num)/(den*den) * a_dvalues[c] * weight;
        break;
      default:
        tmp -= (RealType(2)*den*binaryLabel - RealType(p)*pow(a_values[c], p-1)*num)/(den*den) * a_dvalues[c] * weight;
        break;
      }

    }

    tmp *= scale;
    d_inDataGrad[(b*i64NumChannels + 0)*i64InnerDataNum + i] = tmp;
  }
}

__global__ void DiceLossBackwardKernelDataHalf(const __half *d_inData, const int64_t *d_i64InMask, const float *d_inWeight, const float *d_num, const float *d_den, const float *d_outLossGrad, __half *d_inDataGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, int p, ReductionType eReduction, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  extern __shared__ uint8_t s[]; 

  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  // Work memory for softmax()
  __half * const d_values = ((__half *)s) + i64NumChannels*(threadIdx.y * blockDim.x + threadIdx.x);

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label == i64IgnoreLabel)
      return;

    float scale = 1.0f;

    switch (eReduction) {
    case NoneReduction:
      scale = d_outLossGrad[b];
      break;
    case MeanReduction:
      scale = *d_outLossGrad / (float)(i64BatchSize);
      break;
    case SumReduction:
    case BatchDiceReduction: // Fall through
      scale = *d_outLossGrad;
      break;
    default:
      break;
    }

    if (eReduction != BatchDiceReduction) {
      d_num += b*i64NumChannels;
      d_den += b*i64NumChannels;
    }

    softmax(d_values, d_inData + (b*i64NumChannels + 0)*i64InnerDataNum + i, i64NumChannels, i64InnerDataNum);

    for (int64_t c = 0; c < i64NumChannels; ++c) {
      const float weight = d_inWeight != nullptr ? d_inWeight[c] : 1.0f/(float)i64NumChannels;

      float tmp = 0.0f;
      for (int64_t c2 = 0; c2 < i64NumChannels; ++c2) {
        // NOTE: num and den may overflow __half, we just do this computation in 32 bits
        if (c2 == i64IgnoreChannel)
          continue;

        const float num = d_num[c2];
        const float den = d_den[c2];

        const float binaryLabel = (c2 == i64Label ? 1.0f : 0.0f);
        const float delta = (c == c2 ? 1.0f : 0.0f);
        const float value = __half2float(d_values[c2]);

        if (den == 0.0f)
          continue;

        switch(p) {
        case 1:
          tmp -= (2.0f*den*binaryLabel - num)/(den*den) * (delta - value); // Missing d_values[c] is moved below
          break;
        default:
          tmp -= (2.0f*den*binaryLabel - p*num*pow(value, p-1))/(den*den) * (delta - value); 
          break;
        }
      }

      tmp *= scale * weight * __half2float(d_values[c]);
      d_inDataGrad[(b*i64NumChannels + c)*i64InnerDataNum + i] = __float2half(tmp);
    }
  }
}

__global__ void BinaryDiceLossBackwardKernelDataHalf(const __half *d_inData, const int64_t *d_i64InMask, const float *d_inWeight, const float *d_num, const float *d_den, const float *d_outLossGrad, __half *d_inDataGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, int p, ReductionType eReduction, int64_t i64BatchSize, int64_t i64NumChannels, int64_t i64InnerDataNum) {
  const int64_t b = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (b < i64BatchSize && i < i64InnerDataNum) {
    const int64_t i64Label = d_i64InMask[b*i64InnerDataNum + i];

    if (i64Label == i64IgnoreLabel)
      return;

    float scale = 1.0f;

    switch (eReduction) {
    case NoneReduction:
      scale = d_outLossGrad[b];
      break;
    case MeanReduction:
      scale = *d_outLossGrad / (float)(i64BatchSize);
      break;
    case SumReduction:
    case BatchDiceReduction: // Fall through
      scale = *d_outLossGrad;
      break;
    default:
      break;
    }

    if (eReduction != BatchDiceReduction) {
      d_num += b*2;
      d_den += b*2;
    }

    __half s =  __ushort_as_half(0);
    const __half ds = dsigmoid(d_inData[(b*i64NumChannels + 0)*i64InnerDataNum + i], s);
    const float a_values[2] = { 1.0f - __half2float(s), __half2float(s) };
    const float a_dvalues[2] = { -__half2float(ds), __half2float(ds) };

    float tmp = 0.0f;

    for (int64_t c = 0; c < 2; ++c) {
      const float weight = d_inWeight != nullptr ? d_inWeight[c] : 0.5f;

      if (c == i64IgnoreChannel)
        continue;


      const float num = d_num[c];
      const float den = d_den[c];
      const float binaryLabel = (c == i64Label ? 1.0f : 0.0f);

      if (den == 0.0f)
        continue;
      
      switch (p) {
      case 1:
        tmp -= (2.0f*den*binaryLabel - num)/(den*den) * a_dvalues[c] * weight;
        break;
      default:
        tmp -= (2.0f*den*binaryLabel - p*pow(a_values[c], p-1)*num)/(den*den) * a_dvalues[c] * weight;
        break;
      }
    }

    tmp *= scale;
    d_inDataGrad[(b*i64NumChannels + 0)*i64InnerDataNum + i] = __float2half(tmp);
  }
}

} // end anonymous namespace

template<typename RealType>
torch::Tensor binarydiceloss_gpu_forward(torch::Tensor inData, torch::Tensor inMask, torch::Tensor inWeight, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction) {
  // NOTE: kInt64 should match torch.long
  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return torch::Tensor();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes()[1] != 1 || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  if (inWeight.numel() > 0 && (inWeight.dim() != 1 || inWeight.numel() != 2))
    return torch::Tensor();

  int64_t i64InnerDataNum = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), IntArrayRef::value_type(1), std::multiplies<IntArrayRef::value_type>());
  }

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  const dim3 threadsPerBlock(64,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y - 1)/threadsPerBlock.y);

  const RealType * const d_inData = inData.data_ptr<RealType>();
  const int64_t * const d_i64InMask = inMask.data_ptr<int64_t>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, 2, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return torch::Tensor();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, 2 }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, 2 }, clOptions);

  {
    RealType * const d_num = num.data_ptr<RealType>();
    RealType * const d_den = den.data_ptr<RealType>();

    BinaryDiceLossForwardKernel<RealType><<<numBlocks, threadsPerBlock>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, p, i64BatchSize, i64NumChannels, i64InnerDataNum);
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);
  }

  num *= RealType(2);
  num += smooth;
  den += smooth;

  torch::Tensor outLoss = torch::zeros_like(num);

  {    // Over channels instead of InnerDataNum
    const dim3 numBlocks((2 + threadsPerBlock.x-1)/threadsPerBlock.x, (outLoss.sizes()[0] + threadsPerBlock.y - 1)/threadsPerBlock.y);
    RealType * const d_outLoss = outLoss.data_ptr<RealType>();

    const RealType * const d_num = num.data_ptr<RealType>();
    const RealType * const d_den = den.data_ptr<RealType>();

    // This kernel deals with corner case
    DiceLossForwardKernel2<RealType><<<numBlocks, threadsPerBlock>>>(d_num, d_den, d_outLoss, outLoss.sizes()[0], 2);
  }

  if (inWeight.numel() > 0) {
    outLoss *= inWeight;
    outLoss = outLoss.sum(IntArrayRef(1));
  }
  else
    outLoss = outLoss.mean(IntArrayRef(1));

  switch (eReduction) {
  case NoneReduction:
  case BatchDiceReduction: // Fall through
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
torch::Tensor diceloss_gpu_forward(torch::Tensor inData, torch::Tensor inMask, torch::Tensor inWeight, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction) {
  // NOTE: kInt64 should match torch.long
  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return torch::Tensor();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes()[1] < 2 || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  if (inWeight.numel() > 0 && (inWeight.dim() != 1 || inWeight.numel() != i64NumChannels))
    return torch::Tensor();

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

  {
    RealType * const d_num = num.data_ptr<RealType>();
    RealType * const d_den = den.data_ptr<RealType>();

    DiceLossForwardKernel<RealType><<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, p, i64BatchSize, i64NumChannels, i64InnerDataNum);
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);
  }

  num *= RealType(2);
  num += smooth;
  den += smooth;

  torch::Tensor outLoss = torch::zeros_like(num);

  {    // Over channels instead of InnerDataNum
    const dim3 numBlocks((i64NumChannels + threadsPerBlock.x-1)/threadsPerBlock.x, (outLoss.sizes()[0] + threadsPerBlock.y - 1)/threadsPerBlock.y);
    RealType * const d_outLoss = outLoss.data_ptr<RealType>();

    const RealType * const d_num = num.data_ptr<RealType>();
    const RealType * const d_den = den.data_ptr<RealType>();

    // This kernel deals with corner case
    DiceLossForwardKernel2<RealType><<<numBlocks, threadsPerBlock>>>(d_num, d_den, d_outLoss, outLoss.sizes()[0], i64NumChannels);
  }

  if (inWeight.numel() > 0) {
    outLoss *= inWeight;
    outLoss = outLoss.sum(IntArrayRef(1));
  }
  else
    outLoss = outLoss.mean(IntArrayRef(1));

  switch (eReduction) {
  case NoneReduction:
  case BatchDiceReduction: // Fall through
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
std::vector<torch::Tensor> binarydiceloss_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor inWeight, bool bInWeightGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction) {
  if (bInMaskGrad || bInWeightGrad) // These are never differentiable!
    return std::vector<torch::Tensor>(); 

  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return std::vector<torch::Tensor>();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes()[1] != 1 || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return std::vector<torch::Tensor>();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  if (inWeight.numel() > 0 && (inWeight.dim() != 1 || inWeight.numel() != 2))
    return std::vector<torch::Tensor>();

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
  case BatchDiceReduction: // Fall through
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

  const RealType * const d_inData = inData.data_ptr<RealType>();
  const int64_t * const d_i64InMask = inMask.data_ptr<int64_t>();
  const RealType * const d_inWeight = inWeight.numel() > 0 ? inWeight.data_ptr<RealType>() : nullptr;
  const RealType * const d_outLossGrad = outLossGrad.data_ptr<RealType>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, 2, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return std::vector<torch::Tensor>();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, 2 }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, 2 }, clOptions);

  {
    RealType * const d_num = num.data_ptr<RealType>();
    RealType * const d_den = den.data_ptr<RealType>();

    BinaryDiceLossForwardKernel<RealType><<<numBlocks, threadsPerBlock>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, p, i64BatchSize, i64NumChannels, i64InnerDataNum);
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);
  }

  num *= RealType(2);
  num += smooth;
  den += smooth;

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const d_inDataGrad = inDataGrad.data_ptr<RealType>();

    const RealType * const d_num = num.data_ptr<RealType>();
    const RealType * const d_den = den.data_ptr<RealType>();

    BinaryDiceLossBackwardKernelData<RealType><<<numBlocks, threadsPerBlock>>>(d_inData, d_i64InMask, d_inWeight, d_num, d_den, d_outLossGrad, d_inDataGrad, i64IgnoreChannel, i64IgnoreLabel, p, eReduction, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[0] = inDataGrad;
  }

  return vGradTensors;
}

template<typename RealType>
std::vector<torch::Tensor> diceloss_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor inWeight, bool bInWeightGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction) {
  if (bInMaskGrad || bInWeightGrad) // These are never differentiable!
    return std::vector<torch::Tensor>(); 

  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return std::vector<torch::Tensor>();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes()[1] < 2 || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return std::vector<torch::Tensor>();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  if (inWeight.numel() > 0 && (inWeight.dim() != 1 || inWeight.numel() != i64NumChannels))
    return std::vector<torch::Tensor>();

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
  case BatchDiceReduction: // Fall through
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
  const RealType * const d_inWeight = inWeight.numel() > 0 ? inWeight.data_ptr<RealType>() : nullptr;
  const RealType * const d_outLossGrad = outLossGrad.data_ptr<RealType>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, i64NumChannels, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return std::vector<torch::Tensor>();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);

  {
    RealType * const d_num = num.data_ptr<RealType>();
    RealType * const d_den = den.data_ptr<RealType>();

    DiceLossForwardKernel<RealType><<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, p, i64BatchSize, i64NumChannels, i64InnerDataNum);
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);
  }

  num *= RealType(2);
  num += smooth;
  den += smooth;

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const d_inDataGrad = inDataGrad.data_ptr<RealType>();

    const RealType * const d_num = num.data_ptr<RealType>();
    const RealType * const d_den = den.data_ptr<RealType>();

    DiceLossBackwardKernelData<RealType><<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_inWeight, d_num, d_den, d_outLossGrad, d_inDataGrad, i64IgnoreChannel, i64IgnoreLabel, p, eReduction, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[0] = inDataGrad;
  }

  return vGradTensors;
}

torch::Tensor binarydiceloss_gpu_forward_half(torch::Tensor inData, torch::Tensor inMask, torch::Tensor inWeight, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, float smooth, int p, ReductionType eReduction) {
  // NOTE: kInt64 should match torch.long
  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return torch::Tensor();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes()[1] != 1 || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return torch::Tensor();


  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  if (inWeight.numel() > 0 && (inWeight.dim() != 1 || inWeight.numel() != 2))
    return torch::Tensor();

  int64_t i64InnerDataNum = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), IntArrayRef::value_type(1), std::multiplies<IntArrayRef::value_type>());
  }
  
  auto clOptions = torch::TensorOptions().dtype(torch::kFloat32).device(inData.device());

  const dim3 threadsPerBlock(64,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y - 1)/threadsPerBlock.y);

  const __half * const d_inData = (__half *)inData.data_ptr<torch::Half>();
  const int64_t * const d_i64InMask = inMask.data_ptr<int64_t>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, 2, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return torch::Tensor();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, 2 }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, 2 }, clOptions);

  {
    float * const d_num = num.data_ptr<float>();
    float * const d_den = den.data_ptr<float>();

    BinaryDiceLossForwardKernelHalf<<<numBlocks, threadsPerBlock>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, p, i64BatchSize, i64NumChannels, i64InnerDataNum);
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);
  }

  num *= 2.0f;
  num += smooth;
  den += smooth;

  torch::Tensor outLoss = torch::zeros_like(num);

  {    // Over channels instead of InnerDataNum
    const dim3 numBlocks((2 + threadsPerBlock.x-1)/threadsPerBlock.x, (outLoss.sizes()[0] + threadsPerBlock.y - 1)/threadsPerBlock.y);
    float * const d_outLoss = outLoss.data_ptr<float>();

    const float * const d_num = num.data_ptr<float>();
    const float * const d_den = den.data_ptr<float>();

    // This kernel deals with corner case
    DiceLossForwardKernel2<float><<<numBlocks, threadsPerBlock>>>(d_num, d_den, d_outLoss, outLoss.sizes()[0], 2);
  }

  if (inWeight.numel() > 0) {
    outLoss *= inWeight;
    outLoss = outLoss.sum(IntArrayRef(1));
  }
  else
    outLoss = outLoss.mean(IntArrayRef(1));

  switch (eReduction) {
  case NoneReduction:
  case BatchDiceReduction: // Fall through
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

torch::Tensor diceloss_gpu_forward_half(torch::Tensor inData, torch::Tensor inMask, torch::Tensor inWeight, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, float smooth, int p, ReductionType eReduction) {
  // NOTE: kInt64 should match torch.long
  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return torch::Tensor();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes()[1] < 2 || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return torch::Tensor();


  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  if (inWeight.numel() > 0 && (inWeight.dim() != 1 || inWeight.numel() != i64NumChannels))
    return torch::Tensor();

  int64_t i64InnerDataNum = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), IntArrayRef::value_type(1), std::multiplies<IntArrayRef::value_type>());
  }
  
  auto clOptions = torch::TensorOptions().dtype(torch::kFloat32).device(inData.device());

  const dim3 threadsPerBlock(64,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y - 1)/threadsPerBlock.y);
  const size_t sharedMem = i64NumChannels*threadsPerBlock.x*threadsPerBlock.y*sizeof(__half);

  const __half * const d_inData = (__half *)inData.data_ptr<torch::Half>();
  const int64_t * const d_i64InMask = inMask.data_ptr<int64_t>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, i64NumChannels, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return torch::Tensor();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);

  {
    float * const d_num = num.data_ptr<float>();
    float * const d_den = den.data_ptr<float>();

    DiceLossForwardKernelHalf<<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, p, i64BatchSize, i64NumChannels, i64InnerDataNum);
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);
  }

  num *= 2.0f;
  num += smooth;
  den += smooth;

  torch::Tensor outLoss = torch::zeros_like(num);

  {    // Over channels instead of InnerDataNum
    const dim3 numBlocks((i64NumChannels + threadsPerBlock.x-1)/threadsPerBlock.x, (outLoss.sizes()[0] + threadsPerBlock.y - 1)/threadsPerBlock.y);
    float * const d_outLoss = outLoss.data_ptr<float>();

    const float * const d_num = num.data_ptr<float>();
    const float * const d_den = den.data_ptr<float>();

    // This kernel deals with corner case
    DiceLossForwardKernel2<float><<<numBlocks, threadsPerBlock>>>(d_num, d_den, d_outLoss, outLoss.sizes()[0], i64NumChannels);
  }

  if (inWeight.numel() > 0) {
    outLoss *= inWeight;
    outLoss = outLoss.sum(IntArrayRef(1));
  }
  else
    outLoss = outLoss.mean(IntArrayRef(1));

  switch (eReduction) {
  case NoneReduction:
  case BatchDiceReduction: // Fall through
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

std::vector<torch::Tensor> binarydiceloss_gpu_backward_half(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor inWeight, bool bInWeightGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, float smooth, int p, ReductionType eReduction) {
  if (bInMaskGrad) // This is never differentiable!
    return std::vector<torch::Tensor>(); 

  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return std::vector<torch::Tensor>();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes()[1] != 1 || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return std::vector<torch::Tensor>();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  if (inWeight.numel() > 0 && (inWeight.dim() != 1 || inWeight.numel() != 2))
    return std::vector<torch::Tensor>();

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
  case BatchDiceReduction: // Fall through
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

  auto clOptions = torch::TensorOptions().dtype(torch::kFloat32).device(inData.device());

  const dim3 threadsPerBlock(64,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y - 1)/threadsPerBlock.y);

  const __half * const d_inData = (__half *)inData.data_ptr<torch::Half>();
  const int64_t * const d_i64InMask = inMask.data_ptr<int64_t>();
  const float * const d_inWeight = inWeight.numel() > 0 ? inWeight.data_ptr<float>() : nullptr;
  const float * const d_outLossGrad = outLossGrad.data_ptr<float>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, 2, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return std::vector<torch::Tensor>();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, 2 }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, 2 }, clOptions);

  {
    float * const d_num = num.data_ptr<float>();
    float * const d_den = den.data_ptr<float>();

    BinaryDiceLossForwardKernelHalf<<<numBlocks, threadsPerBlock>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, p, i64BatchSize, i64NumChannels, i64InnerDataNum);
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);
  }

  num *= 2.0f;
  num += smooth;
  den += smooth;

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    __half * const d_inDataGrad = (__half *)inDataGrad.data_ptr<torch::Half>();

    const float * const d_num = num.data_ptr<float>();
    const float * const d_den = den.data_ptr<float>();

    BinaryDiceLossBackwardKernelDataHalf<<<numBlocks, threadsPerBlock>>>(d_inData, d_i64InMask, d_inWeight, d_num, d_den, d_outLossGrad, d_inDataGrad, i64IgnoreChannel, i64IgnoreLabel, p, eReduction, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[0] = inDataGrad;
  }

  return vGradTensors;
}

std::vector<torch::Tensor> diceloss_gpu_backward_half(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor inWeight, bool bInWeightGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, float smooth, int p, ReductionType eReduction) {
  if (bInMaskGrad) // This is never differentiable!
    return std::vector<torch::Tensor>(); 

  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return std::vector<torch::Tensor>();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes()[1] < 2 || inData.sizes().slice(2) != inMask.sizes().slice(1))
    return std::vector<torch::Tensor>();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];

  if (inWeight.numel() > 0 && (inWeight.dim() != 1 || inWeight.numel() != i64NumChannels))
    return std::vector<torch::Tensor>();

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
  case BatchDiceReduction: // Fall through
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

  auto clOptions = torch::TensorOptions().dtype(torch::kFloat32).device(inData.device());

  const dim3 threadsPerBlock(64,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y - 1)/threadsPerBlock.y);
  const size_t sharedMem = i64NumChannels*threadsPerBlock.x*threadsPerBlock.y*sizeof(__half);

  const __half * const d_inData = (__half *)inData.data_ptr<torch::Half>();
  const int64_t * const d_i64InMask = inMask.data_ptr<int64_t>();
  const float * const d_inWeight = inWeight.numel() > 0 ? inWeight.data_ptr<float>() : nullptr;
  const float * const d_outLossGrad = outLossGrad.data_ptr<float>();

  torch::Tensor invalidFound = torch::scalar_tensor(false, torch::TensorOptions().dtype(torch::kBool).device(inData.device()));
  CheckForInvalidLabels<<<numBlocks, threadsPerBlock>>>(d_i64InMask, invalidFound.data_ptr<bool>(), i64IgnoreLabel, i64BatchSize, i64NumChannels, i64InnerDataNum);

  if (invalidFound.to(torch::kCPU).item<bool>()) {
    std::cerr << "Error: Invalid label encountered in ground truth mask!" << std::endl;
    return std::vector<torch::Tensor>();
  }

  torch::Tensor num = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);

  {
    float * const d_num = num.data_ptr<float>();
    float * const d_den = den.data_ptr<float>();

    DiceLossForwardKernelHalf<<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_num, d_den, i64IgnoreChannel, i64IgnoreLabel, p, i64BatchSize, i64NumChannels, i64InnerDataNum);
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);
  }

  num *= 2.0f;
  num += smooth;
  den += smooth;

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    __half * const d_inDataGrad = (__half *)inDataGrad.data_ptr<torch::Half>();

    const float * const d_num = num.data_ptr<float>();
    const float * const d_den = den.data_ptr<float>();

    DiceLossBackwardKernelDataHalf<<<numBlocks, threadsPerBlock, sharedMem>>>(d_inData, d_i64InMask, d_inWeight, d_num, d_den, d_outLossGrad, d_inDataGrad, i64IgnoreChannel, i64IgnoreLabel, p, eReduction, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[0] = inDataGrad;
  }

  return vGradTensors;
}

template torch::Tensor binarydiceloss_gpu_forward<float>(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t i64IgnoreLabel, const float &, int, ReductionType);
template torch::Tensor binarydiceloss_gpu_forward<double>(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t i64IgnoreLabel, const double &, int, ReductionType);

template torch::Tensor diceloss_gpu_forward<float>(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t i64IgnoreLabel, const float &, int, ReductionType);
template torch::Tensor diceloss_gpu_forward<double>(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t i64IgnoreLabel, const double &, int, ReductionType);

template std::vector<torch::Tensor> binarydiceloss_gpu_backward<float>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, int64_t, int64_t, const float &, int, ReductionType);
template std::vector<torch::Tensor> binarydiceloss_gpu_backward<double>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, int64_t, int64_t, const double &, int, ReductionType);

template std::vector<torch::Tensor> diceloss_gpu_backward<float>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, int64_t, int64_t, const float &, int, ReductionType);
template std::vector<torch::Tensor> diceloss_gpu_backward<double>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, int64_t, int64_t, const double &, int, ReductionType);

