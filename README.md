# Implementation of George Hinton's Forward-Forward algorithm
[The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)

![Acc](./stuff/acc.png "Accuracy")

## Differences
|             | Original paper                                             | Other implementations                                                | This        |
| ----------- | ---------------------------------------------------------- | -------------------------------------------------------------------- | ----------- |
| Metric      | $\sigma = \sum\limits_{i=1}^{N}y^2$ <br> sum of neuron activities   | $\mu = \frac{1}{N}\sum\limits_{i=1}^{N}y^2$ <br> mean of neuron activities | $l2 = \sqrt{\sum{y^2}}$ <br> L2 of neuron activities, *($y>=0$ due to ReLU activation)*
| Loss        | $\frac{1}{1+e^{-\delta}}$ aka sigmoid function, <br> where $\delta = \pm threshold \mp \sigma $  | $ln(1 + e^{\delta})$, <br> where $\delta=\pm threshold \mp \mu$ | $ln(1+cosh(\delta)+\delta)$, <br> where $\delta = \pm threshold \mp l2$
| Samples     |                                                            | Negative samples added to loss function via concat | Negative samples' loss computed through *states* passed along images and labels  
| Goodness    | Measured across all layers                                 | Measured across all layers                        | Measured at last layer 

I guess reason why everybody else is using $\mu$ to minimize $log$ is because large values can produce $inf$ in $e^{x}$. Still, this implementation allows to use original sigmoid loss (commented in loss function for FFConv and FFLinear layers). Beside changing loss fn, you'll need to change metric, optimizer and increase batch size to 640 samples. I was unable to achieve generalization via first convolutional layer when using sigmoid loss, so you'll need to switch to pure linear model.

Use __FFLinearModel__ from _libs.models_ for the sequence of linear layers.
Use __FFConvModel__ for a more traditional mix of convolutions and linear layers.

## Loss
![Loss](./stuff/loss.png "Loss")