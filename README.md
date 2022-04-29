# diceloss_cpp
Memory-efficient DiceLoss for PyTorch

# Caution!
Be careful with empty segmentation masks (especially if you are ignoring the background channel). The dice loss gradient is not necessarily 0 and can lead to head-scratching moments! Perhaps `p = 2` and `smooth = 1` are safer options for this corner case. This is what other dice loss implementations generally use.

# Introduction
Some of the dice loss implementations I've seen calculate softmax and one-hot encoded masks. This is not a big deal in 2D, but in 3D, this is extremely wasteful in memory. If you have batch size B and K classes, the one-hot mask will require B * K * H * W * D in memory and the softmax will require twice that (one for the gradient). So my workaround is to... not store one-hot encoded masks or softmax and instead calculate everything on the fly in the dice loss.

The DiceLoss in this repository calculates the multi-class dice loss (average of binary dice losses over classes/channels) and fuses one-hot and softmax into the dice loss calculation so that you do not need to store one-hot encoded masks or softmax. Code for both CPU and GPU have been implemented and tested. It's almost certaintly not computationally optimal! But at least you can fit more on the GPU!

This is still experimental. Beware of bugs! Though it's chewing through KITS19 data for me just fine!

# Compiling diceloss_cpp
The diceloss_cpp PyTorch extension can be compiled using setup.py
```shell
python setup.py build
python setup.py install
```

# Usage
Once compiled and installed, you should be able to do something like this:
```py
from DiceLoss import DiceLoss

dice = DiceLoss(ignore_channel = -1, ignore_label = -1, smooth = 1e-3, p = 1, reduction = "mean")
x = torch.rand([8, 5, 100, 100, 100]).cuda()
y = torch.randint(size=[8, 100, 100, 100], low=0, high=6).type(torch.long).cuda()

loss = dice(x,y)
loss.backward()
```
**NOTE**: `p` must be an integer.

## Inputs
* `x` -- [ BatchSize, Channels, d1, d2, ... ] (torch.float32 or torch.float64)
* `y`-- [ BatchSize, d1, d2, ... ] (torch.long)

## Outputs
* `reduction` = "mean"/"sum" -- scalar
* `reduction` = "none" -- [ BatchSize ]

## `ignore_channel`
This ignores computing the binary dice loss along one of the input channels. If `ignore_channel` is not one of the integers in the range [0, Channels), then this option has no effect on dice loss calculation (binary dice loss will be calculated along all channels). Binary dice loss for other channels will be computed normally even when encountering mask labels of `ignore_channel`. This is useful for ignoring the background label.

## `ignore_label`
In contrast to `ignore_channel`, this ignores any computation related to **all** binary dice losses over all channels whenever the mask label is `ignore_label`. This is useful for ignoring *don't-care* or *unknown* regions of an image. There will be no loss or gradient contribution in regions of the mask with label `ignore_label`. The `ignore_label` can be any integer. If the mask has no `ignore_label` labels, this option has no effect on dice loss calculation.

## `smooth`, `p`
These are terms in the dice calculation given below
```
dice(x,y) = 1 - (2*x'*y + smooth)/(|x|_p^p + |y|_p^p + smooth)
```
where `|x|_p` is the p-norm of `x`.


## `reduction`
This can be "mean", "sum" or "none".
* "mean" -- calculates the mean dice loss over the batch (each individual batch instance being an average over class labels).
* "sum" -- calculates the sum of dice losses over the batch.
* "none" -- returns a list of losses (one loss for each batch instance).

