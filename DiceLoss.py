# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# April 2022
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.autograd
import diceloss_cpp

def _is_deterministic():
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        return torch.are_deterministic_algorithms_enabled()
    elif hasattr(torch, "is_deterministic"):
        return torch.is_deterministic()

    raise RuntimeError("Unable to query torch deterministic mode.")

class _DiceLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inMask, ignoreChannel, ignoreLabel, smooth, p, reduction):
        if _is_deterministic() and inData.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of diceloss forward on GPUs.")

        ctx.save_for_backward(inData, inMask)

        ctx.ignoreChannel = ignoreChannel
        ctx.ignoreLabel = ignoreLabel
        ctx.smooth = smooth
        ctx.p = p
        ctx.reduction = reduction

        return diceloss_cpp.diceloss_forward(inData.contiguous(), inMask.contiguous(), ctx.ignoreChannel, ctx.ignoreLabel, ctx.smooth, ctx.p, ctx.reduction)

    @staticmethod
    def backward(ctx, outLossGrad):
        if _is_deterministic() and outLossGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation of diceloss on GPUs.")

        inData, inMask = ctx.saved_tensors

        inDataGrad, inMaskGrad = diceloss_cpp.diceloss_backward(inData.contiguous(), ctx.needs_input_grad[0], inMask.contiguous(), ctx.needs_input_grad[1], outLossGrad.contiguous(), ctx.ignoreChannel, ctx.ignoreLabel, ctx.smooth, ctx.p, ctx.reduction)

        return inDataGrad, inMaskGrad, None, None, None, None, None

class DiceLoss(nn.Module):
    supported_reductions = { "none", "mean", "sum" }

    def __init__(self, ignore_channel=-1, ignore_label=-1, smooth=1e-3, p=1, reduction="mean"):
        super().__init__()

        if reduction not in DiceLoss.supported_reductions:
            raise RuntimeError(f"Unsupported reduction '{reduction}'. Supported reductions are: {DiceLoss.supported_reductions}.")

        self.ignore_channel = ignore_channel
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, x, y):
        return _DiceLoss.apply(x, y, self.ignore_channel, self.ignore_label, self.smooth, self.p, self.reduction)

