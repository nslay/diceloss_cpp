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

// One-hot encoding and softmax are wasteful for 3D. Implement dice loss in a native language to avoid that!

#pragma once

#ifndef DICELOSS_H
#define DICELOSS_H

#include <string>

// This enum is the ONLY reason this file exists!
enum ReductionType { UnknownReduction = -1, NoneReduction, MeanReduction, SumReduction, BatchDiceReduction };

ReductionType GetReductionByName(std::string strReduction);

template<typename RealType>
torch::Tensor diceloss_cpu_forward(torch::Tensor inData, torch::Tensor inMask, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction);

template<typename RealType>
std::vector<torch::Tensor> diceloss_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction);

template<typename RealType>
torch::Tensor diceloss_gpu_forward(torch::Tensor inData, torch::Tensor inMask, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction);

template<typename RealType>
std::vector<torch::Tensor> diceloss_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inMask, bool bInMaskGrad, torch::Tensor outLossGrad, int64_t i64IgnoreChannel, int64_t i64IgnoreLabel, const RealType &smooth, int p, ReductionType eReduction);

#endif // !DICELOSS_H


