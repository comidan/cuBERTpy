# cuBERTpy
This repository is proposing a fully working BERT architecture completely implemented from scratch in CuPy for Text Classification and completely equivalent to the one proposed by HuggingFace.
You'll be able to better understand the behind the scenes of these complex models by looking at a scratch implementation using a mathematical library like CuPy and also you'll find the code to be better managed than what it's proposed as a single file in other libraries implementations.

I used CuPy, which is completely equivalent to NumPy, just to add GPU support for faster inference even if it won't be as fast as the more optimized frameworks of Pytorch and Tensorflow.

I also added a class in order to load and convert weights from a Pytorch saved model and load them in the CuPy tensors of the model.

This work is the product of an other research project I've been working on.

For any question or issue please contact me!

Thanks to [CuPy](https://github.com/cupy/cupy), [Pytorch](https://github.com/pytorch/pytorch) and [HuggingFace](https://github.com/huggingface/transformers)!
