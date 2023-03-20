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

// One-hot encoding is wasteful for 3D. Implement diceloss in a native language to avoid that!

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
#include "diceloss.h"

typedef c10::IntArrayRef IntArrayRef;

namespace {

template<typename RealType>
void softmax(RealType *p_outData, const RealType *p_inData, int64_t i64NumChannels, int64_t i64Stride) {
  RealType offset = p_inData[i64Stride*0];
  for (int64_t c = 1; c < i64NumChannels; ++c) {
    if (offset < p_inData[i64Stride*c])
      offset = p_inData[i64Stride*c];
  }

  RealType norm = RealType(0);
  for (int64_t c = 0; c < i64NumChannels; ++c) {
    p_outData[c] = std::exp(p_inData[i64Stride*c] - offset);
    norm += p_outData[c];
  }

  for (int64_t c = 0; c < i64NumChannels; ++c)
    p_outData[c] /= norm;
}

} // end anonymous namespace

ReductionType GetReductionByName(std::string strReduction) {
  std::transform(strReduction.begin(), strReduction.end(), strReduction.begin(), 
    [](unsigned char c) -> char {
      return std::tolower(c);
    });

  if (strReduction == "none")
    return NoneReduction;
  else if (strReduction == "mean")
    return MeanReduction;
  else if (strReduction == "sum")
    return SumReduction;
  else if (strReduction == "batch")
    return BatchDiceReduction;

  return UnknownReduction;
}

template<typename RealType>
torch::Tensor diceloss_cpu_forward(torch::Tensor inData, torch::Tensor inMask, torch::Tensor inWeight, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction) {
  // NOTE: kInt64 should match torch.long
  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return torch::Tensor();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes().slice(2) != inMask.sizes().slice(1))
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

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const int64_t * const p_i64InMask = inMask.data_ptr<int64_t>();

  torch::Tensor num = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor values = torch::empty({ i64NumChannels }, clOptions);

  RealType *p_num = num.data_ptr<RealType>();
  RealType *p_den = den.data_ptr<RealType>();
  RealType * const p_values = values.data_ptr<RealType>();

  for (int64_t b = 0; b < i64BatchSize; ++b) {
    for (int64_t i = 0; i < i64InnerDataNum; ++i) {
      const int64_t i64Label = p_i64InMask[b*i64InnerDataNum + i];

      if (i64Label == i64IgnoreLabel)
        continue;
 
      if (i64Label < 0 || i64Label >= i64NumChannels) {
        std::cerr << "Error: Invalid label encountered in ground truth mask (" << i64Label << ")!" << std::endl;
        return torch::Tensor();
      }

      softmax(p_values, p_inData + (b*i64NumChannels + 0)*i64InnerDataNum + i, i64NumChannels, i64InnerDataNum);

      for (int64_t c = 0; c < i64NumChannels; ++c) {
        if (c == i64IgnoreChannel)
          continue;

        const RealType value = p_values[c];
        const RealType binaryLabel = RealType(i64Label == c ? 1 : 0);
        p_num[b*i64NumChannels + c] += value*binaryLabel;
        p_den[b*i64NumChannels + c] += std::pow(value, p) + binaryLabel; // No reason to calculate std::pow(binaryLabel, p)
      }
    }
  }

  if (eReduction == BatchDiceReduction) {
    num = num.sum(IntArrayRef(0), true);
    den = den.sum(IntArrayRef(0), true);

    p_num = num.data_ptr<RealType>();
    p_den = den.data_ptr<RealType>();
  }

  num *= RealType(2);
  num += smooth;
  den += smooth;

  torch::Tensor outLoss = torch::zeros_like(num);
  RealType *p_outLoss = outLoss.data_ptr<RealType>();

  for (int64_t b = 0; b < outLoss.sizes()[0]; ++b) {
    for (int64_t c = 0; c < i64NumChannels; ++c) {
      if (c == i64IgnoreChannel)
        continue;

      const RealType num = p_num[b*i64NumChannels + c];
      const RealType den = p_den[b*i64NumChannels + c];

      // By definition: den >= num and we say DSC = 1 when comparing two empty sets (which are the same)
      if (den == RealType(0))
        continue;
 
      p_outLoss[b*i64NumChannels + c] = RealType(1) - num/den;
    }
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
std::vector<torch::Tensor> diceloss_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor inWeight, bool bInWeightGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction) {
  if (bInMaskGrad || bInWeightGrad) // These are never differentiable!
    return std::vector<torch::Tensor>(); 

  if (inData.dim() < 2 || inData.dim() != inMask.dim()+1 || inMask.scalar_type() != torch::kInt64)
    return std::vector<torch::Tensor>();

  if (inData.sizes()[0] != inMask.sizes()[0] || inData.sizes().slice(2) != inMask.sizes().slice(1))
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

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const int64_t * const p_i64InMask = inMask.data_ptr<int64_t>();
  const RealType * const p_inWeight = inWeight.numel() > 0 ? inWeight.data_ptr<RealType>() : nullptr;
  RealType * const p_outLossGrad = outLossGrad.data_ptr<RealType>();

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor num = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor den = torch::zeros({ i64BatchSize, i64NumChannels }, clOptions);
  torch::Tensor values = torch::empty({ i64NumChannels }, clOptions);

  RealType * const p_num = num.data_ptr<RealType>();
  RealType * const p_den = den.data_ptr<RealType>();
  RealType * const p_values = values.data_ptr<RealType>();

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();

    for (int64_t b = 0; b < i64BatchSize; ++b) {
      for (int64_t i = 0; i < i64InnerDataNum; ++i) {
        const int64_t i64Label = p_i64InMask[b*i64InnerDataNum + i];

        if (i64Label == i64IgnoreLabel)
          continue;
 
        if (i64Label < 0 || i64Label >= i64NumChannels) {
          std::cerr << "Error: Invalid label encountered in ground truth mask (" << i64Label << ")!" << std::endl;
          return std::vector<torch::Tensor>();
        }

        softmax(p_values, p_inData + (b*i64NumChannels + 0)*i64InnerDataNum + i, i64NumChannels, i64InnerDataNum);

        for (int64_t c = 0; c < i64NumChannels; ++c) {
          if (c == i64IgnoreChannel) // No, we need to multiply with the full Jacobian... can't ignore here
            continue;

          const RealType binaryLabel = RealType(i64Label == c ? 1 : 0);
          p_num[b*i64NumChannels + c] += p_values[c]*binaryLabel;
          p_den[b*i64NumChannels + c] += std::pow(p_values[c], p) + binaryLabel; // No reason to calculate std::pow(binaryLabel, p)
        }
      }
    }

    if (eReduction == BatchDiceReduction) {
      num = num.sum(IntArrayRef(0), true);
      den = den.sum(IntArrayRef(0), true);

      // XXX: These pointers are invalid!
      //p_num = num.data_ptr<RealType>();
      //p_den = den.data_ptr<RealType>();
    }

    num *= RealType(2);
    num += smooth;
    den += smooth;

    for (int64_t b = 0; b < i64BatchSize; ++b) {
      const RealType *p_num = num.data_ptr<RealType>();
      const RealType *p_den = den.data_ptr<RealType>();

      if (eReduction != BatchDiceReduction) {
        p_num += b*i64NumChannels;
        p_den += b*i64NumChannels;
      }

      RealType scale = RealType(1);

      switch (eReduction) {
      case NoneReduction:
        scale = p_outLossGrad[b];
        break;
      case MeanReduction:
        scale = *p_outLossGrad / RealType(i64BatchSize);
        break;
      case SumReduction:
      case BatchDiceReduction: // Fall through
        scale = *p_outLossGrad;
        break;
      default:
        break;
      }

      for (int64_t i = 0; i < i64InnerDataNum; ++i) {
        const int64_t i64Label = p_i64InMask[b*i64InnerDataNum + i];

        if (i64Label == i64IgnoreLabel)
          continue;
 
        softmax(p_values, p_inData + (b*i64NumChannels + 0)*i64InnerDataNum + i, i64NumChannels, i64InnerDataNum);

        for (int64_t c = 0; c < i64NumChannels; ++c) {
          const RealType weight = p_inWeight != nullptr ? p_inWeight[c] : RealType(1)/RealType(i64NumChannels);

          // NOTE:  Owing to the Jacobian multiplication, we can't ignore c == ignore_channel here!
          for (int64_t c2 = 0; c2 < i64NumChannels; ++c2) {
            if (c2 == i64IgnoreChannel) // This loss term is discarded (constant), so the partial = 0 for this term
              continue;

            const RealType delta = RealType(c == c2 ? 1 : 0);
            const RealType binaryLabel = RealType(i64Label == c2 ? 1 : 0);

            // Recall den >= num
            if (p_den[c2] == RealType(0)) // If num == 0, the DSC curve is a constant 0, so we say the limiting case has 0 derivative
              continue;

            switch (p) {
            case 1:
              p_inDataGrad[(b*i64NumChannels + c)*i64InnerDataNum + i] -= (RealType(2)*p_den[c2]*binaryLabel - p_num[c2])/(p_den[c2]*p_den[c2]) * (delta - p_values[c2]); // NOTE: Missing p_values[c] factor is moved below
              break;
            default:
              p_inDataGrad[(b*i64NumChannels + c)*i64InnerDataNum + i] -= (RealType(2)*p_den[c2]*binaryLabel - RealType(p)*p_num[c2]*std::pow(p_values[c2],p-1))/(p_den[c2]*p_den[c2]) * (delta - p_values[c2]);
              break;
            }
          }

          p_inDataGrad[(b*i64NumChannels + c)*i64InnerDataNum + i] *= scale * weight * p_values[c];
        }
      }
    }

    vGradTensors[0] = inDataGrad;
  }

  return vGradTensors;
}

#ifndef WITH_CUDA
template<typename RealType>
torch::Tensor diceloss_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, const RealType &, int, ReductionType) {
  return torch::Tensor();
}

template<typename RealType>
std::vector<torch::Tensor> diceloss_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, int64_t, int64_t, const RealType &, int, ReductionType) {
  return std::vector<torch::Tensor>();
}

torch::Tensor diceloss_gpu_forward_half(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, float, int, ReductionType) {
  return torch::Tensor();
}

std::vector<torch::Tensor> diceloss_gpu_backward_half(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, int64_t, int64_t, float, int, ReductionType) {
  return std::vector<torch::Tensor>();
}
#endif // !WITH_CUDA

torch::Tensor diceloss_forward(torch::Tensor inData, torch::Tensor inMask, torch::Tensor inWeight, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, double dSmooth, int p, const std::string &strReduction) {
  if (inMask.scalar_type() != torch::kInt64)
    return torch::Tensor();

  if (inData.device() != inMask.device())
    return torch::Tensor(); 

  if (inWeight.numel() > 0 && inData.device() != inWeight.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inMask.is_contiguous())
    return torch::Tensor();

  if (inWeight.numel() > 0 && !inWeight.is_contiguous())
    return torch::Tensor();

  const ReductionType eReduction = GetReductionByName(strReduction);

  if (eReduction == UnknownReduction)
    return torch::Tensor();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat16:
    {
      if (inWeight.numel() > 0 && inWeight.scalar_type() != torch::kFloat32)
        return torch::Tensor();

      if (inData.is_cuda())
        return diceloss_gpu_forward_half(inData, inMask, inWeight, i64IgnoreChannel, i64IgnoreLabel, (float)dSmooth, p, eReduction);
      else
        return torch::Tensor(); // Not implemented yet
    }
    break;
  case torch::kFloat32:
    {
      if (inWeight.numel() > 0 && inWeight.dtype() != inData.dtype())
        return torch::Tensor();

      if (inData.is_cuda())
        return diceloss_gpu_forward<float>(inData, inMask, inWeight, i64IgnoreChannel, i64IgnoreLabel, (float)dSmooth, p, eReduction);
      else
        return diceloss_cpu_forward<float>(inData, inMask, inWeight, i64IgnoreChannel, i64IgnoreLabel, (float)dSmooth, p, eReduction);
    }
    break;
  case torch::kFloat64:
    {
      if (inWeight.numel() > 0 && inWeight.dtype() != inData.dtype())
        return torch::Tensor();

      if (inData.is_cuda())
        return diceloss_gpu_forward<double>(inData, inMask, inWeight, i64IgnoreChannel, i64IgnoreLabel, dSmooth, p, eReduction);
      else
        return diceloss_cpu_forward<double>(inData, inMask, inWeight, i64IgnoreChannel, i64IgnoreLabel, dSmooth, p, eReduction);
    }
    break;
  default:
    return torch::Tensor();
  }

  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> diceloss_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor inWeight, bool bInWeightGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, double dSmooth, int p, const std::string &strReduction) {
  if (inMask.scalar_type() != torch::kInt64)
    return std::vector<torch::Tensor>();

  if (inData.device() != inMask.device() || inData.device() != outLossGrad.device())
    return std::vector<torch::Tensor>();

  if (inWeight.numel() > 0 && inData.device() != inWeight.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inMask.is_contiguous() || !outLossGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  if (inWeight.numel() > 0 && !inWeight.is_contiguous())
    return std::vector<torch::Tensor>();

  const ReductionType eReduction = GetReductionByName(strReduction);

  if (eReduction == UnknownReduction)
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat16:
    {
      if (inWeight.numel() > 0 && inWeight.scalar_type() != torch::kFloat32)
        return std::vector<torch::Tensor>(); 

      if (outLossGrad.scalar_type() != torch::kFloat32)
        return std::vector<torch::Tensor>(); 
      
      if (inData.is_cuda())
        return diceloss_gpu_backward_half(inData, bInDataGrad, inMask, bInMaskGrad, inWeight, bInWeightGrad, outLossGrad, i64IgnoreChannel, i64IgnoreLabel, (float)dSmooth, p, eReduction);
      else
        return std::vector<torch::Tensor>(); // Not implemented yet
    }
    break;
  case torch::kFloat32:
    {
      if (inData.dtype() != outLossGrad.dtype())
        return std::vector<torch::Tensor>(); 

      if (inWeight.numel() > 0 && inWeight.dtype() != inData.dtype())
        return std::vector<torch::Tensor>(); 

      if (inData.is_cuda())
        return diceloss_gpu_backward<float>(inData, bInDataGrad, inMask, bInMaskGrad, inWeight, bInWeightGrad, outLossGrad, i64IgnoreChannel, i64IgnoreLabel, (float)dSmooth, p, eReduction);
      else
        return diceloss_cpu_backward<float>(inData, bInDataGrad, inMask, bInMaskGrad, inWeight, bInWeightGrad, outLossGrad, i64IgnoreChannel, i64IgnoreLabel, (float)dSmooth, p, eReduction);
    }
    break;
  case torch::kFloat64:
    {
      if (inData.dtype() != outLossGrad.dtype())
        return std::vector<torch::Tensor>(); 

      if (inWeight.numel() > 0 && inWeight.dtype() != inData.dtype())
        return std::vector<torch::Tensor>(); 

      if (inData.is_cuda())
        return diceloss_gpu_backward<double>(inData, bInDataGrad, inMask, bInMaskGrad, inWeight, bInWeightGrad, outLossGrad, i64IgnoreChannel, i64IgnoreLabel, dSmooth, p, eReduction);
      else
        return diceloss_cpu_backward<double>(inData, bInDataGrad, inMask, bInMaskGrad, inWeight, bInWeightGrad, outLossGrad, i64IgnoreChannel, i64IgnoreLabel, dSmooth, p, eReduction);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }

  return std::vector<torch::Tensor>(); // Not reached
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("diceloss_forward", &diceloss_forward, "Dice loss forward without one-hot encodings.");
  m.def("diceloss_backward", &diceloss_backward, "Dice loss backward without one-hot encodings.");
}

