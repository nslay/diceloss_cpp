# diceloss_cpp
Memory-efficient DiceLoss for PyTorch

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

dice = DiceLoss(ignore_channel = -1, ignore_label = -1, smooth = 0, p = 1, reduction = "mean")
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
* `reduction` = "mean"/"sum"/"batch" -- scalar
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
This can be "batch", "mean", "sum" or "none".
* "mean" -- calculates the mean dice loss over the batch (each individual batch instance being an average over class labels).
* "sum" -- calculates the sum of dice losses over the batch.
* "none" -- returns a list of losses (one loss for each batch instance).
* "batch" -- calculates the dice numerator and denominator per-channel over all batch instances.

# Caution!
Be careful with empty segmentation masks (especially if you are ignoring the background channel). The dice loss gradient is not necessarily 0 and can lead to head-scratching moments! Using `smooth=0` can ensure partial derivatives are 0 in these corner cases. Using `reduction="batch"` can help prevent encountering empty segmentation masks since even a single non-empty segmentation mask in a batch will count for the entire batch. Another strategy to deal with batch instances with empty segmentation masks is to zero out the loss for those instances. For example, you can do something like this
```py
ce = nn.CrossEntropyLoss()
dice = DiceLoss(ignore_channel=0, reduction="none") # This reduction returns a list of losses per batch.
...
for xbatch, ybatch in batcher:
  optimizer.zero_grad()
  
  outputs = net(xbatch)
  
  loss1 = ce(outputs, ybatch)
  
  # ybatch is [BatchSize, Height, Width] or [BatchSize, Depth, Height, Width] or etc...
  batchWeight = (ybatch.view([ybatch.shape[0], -1]).max(dim=1)[0] > 0)

  loss2 = dice(outputs, ybatch)
  loss2 = (loss2*batchWeight).mean() # Aggregate with your batch instance weights
  
  loss = loss1 + loss2
  
  loss.backward()
  
  optimizer.step()
```
Forgive me if there are errors. It's just a rough example of what you could do to avoid dice loss issues on empty segmentation masks in your training set.
