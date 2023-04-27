# Implementation of George Hinton's Forward-Forward algorithm
[The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)

![Acc](./stuff/acc.png "Accuracy")

## Differences
- Original paper measures mean of activities of each neuron squared $\mu=\frac{1}{N}\sum\limits_{i=1}^{N}y^2$ for (BxN). This one is using length $\sqrt{\sum{y^2}}$ (for BxCxMxN activity is measured as mean of activities per each channel)
- Original paper normalizes through simple division by L2 length. This one uses $LayerNorm$.
- Other implementations are computing loss as $1+ln(e^{threshold-\mu})$, where $\mu=\frac{1}{N}\sum\limits_{i=1}^{N}y^2$. This one uses $ln(1+cosh(\delta)+\delta)/2.$, where $\delta = threshold - \sqrt{\sum{y^2}}$ or vice versa for negative samples. This loss is designed to provide different targets for length of vectors. Negative vectors' lengths expected to be shorter, positive longer, which is similar to original paper but provides better convergency. In addition, $1+ln(e^x)$ can reach $inf$ for large $x$ at $e^x$ part, using $cosh(x)$ was the reason to prevent such case

Use __FFLinearModel__ from _libs.models_ for the sequence of linear layers.
Use __FFConvModel__ for a more traditional mix of convolutions and linear layers.


## Notes
1. This implementation does not generate pairs of pos and neg samples per each input, instead it modifies pos into neg with a 50% chance. 
2. There are two functions to measure goodness: per layer or sum of all layers combined. Results here computed via last layer only 
3. Model evaluated by processing all possible combinations of input and class labels, peak goodness is considered model's choice. 