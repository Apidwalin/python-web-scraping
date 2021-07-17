##### Copyright 2021 The TensorFlow Authors.


```python
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### Install the TensorFlow Model Garden pip package

*  `tf-models-official` is the stable Model Garden package. Note that it may not include the latest changes in the `tensorflow_models` github repo. To include latest changes, you may install `tf-models-nightly`,
which is the nightly Model Garden package created daily automatically.
*  pip will install all models and dependencies automatically.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/official_models/tutorials/decoding_api_in_tf_nlp.ipynb"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/models/blob/master/official/colab/decoding_api_in_tf_nlp.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/models/blob/master/official/colab/decoding_api_in_tf_nlp.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/models/official/colab/decoding_api_in_tf_nlp.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>


```python
pip install  tf-models-nightly
```


```python
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from official import nlp
from official.nlp.modeling.ops import sampling_module
from official.nlp.modeling.ops import beam_search
```

# Decoding API
This API provides an interface to experiment with different decoding strategies used for auto-regressive models.

1. The following sampling strategies are provided in sampling_module.py, which inherits from the base Decoding class:
  *   [top_p](https://arxiv.org/abs/1904.09751) : [github](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/sampling_module.py#L65) 

      This implementation chooses most probable logits with cumulative probabilities upto top_p.

  *   [top_k](https://arxiv.org/pdf/1805.04833.pdf) : [github](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/sampling_module.py#L48)

      At each timestep, this implementation samples from top-k logits based on their probability distribution

  *   Greedy : [github](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/sampling_module.py#L26)

      This implementation returns the top logits based on probabilities.

2. Beam search is provided in beam_search.py. [github](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/beam_search.py)

      This implementation reduces the risk of missing hidden high probability logits by keeping the most likely num_beams of logits at each time step and eventually choosing the logits that has the overall highest probability.

## Initialize Sampling Module in TF-NLP.


> **symbols_to_logits_fn** : This is a closure implemented by the users of the API. The input to this closure will be  
```
Args:
  1] ids [batch_size, .. (index + 1 or 1 if padded_decode is True)],
  2] index [scalar] : current decoded step,
  3] cache [nested dictionary of tensors].
Returns:
  1] tensor for next-step logits [batch_size, vocab]
  2] the updated_cache [nested dictionary of tensors].
```
This closure calls the model to predict the logits for the 'index+1' step. The cache is used for faster decoding.
Here is a [reference](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/beam_search_test.py#L88) implementation for the above closure.


> **length_normalization_fn** : Closure for returning length normalization parameter.
```
Args: 
  1] length : scalar for decoded step index.
  2] dtype : data-type of output tensor
Returns:
  1] value of length normalization factor.
Example :
  def _length_norm(length, dtype):
    return tf.pow(((5. + tf.cast(length, dtype)) / 6.), 0.0)
```

> **vocab_size** : Output vocabulary size.

> **max_decode_length** : Scalar for total number of decoding steps.

> **eos_id** : Decoding will stop if all output decoded ids in the batch have this ID.

> **padded_decode** : Set this to True if running on TPU. Tensors are padded to max_decoding_length if this is True.

> **top_k** : top_k is enabled if this value is > 1.

> **top_p** : top_p is enabled if this value is > 0 and < 1.0

> **sampling_temperature** : This is used to re-estimate the softmax output. Temperature skews the distribution towards high probability tokens and lowers the mass in tail distribution. Value has to be positive. Low temperature is equivalent to greedy and makes the distribution sharper, while high temperature makes it more flat.

> **enable_greedy** : By default, this is true and greedy decoding is enabled.


# Initialize the Model Hyper-parameters


```python
params = {}
params['num_heads'] = 2
params['num_layers'] = 2
params['batch_size'] = 2
params['n_dims'] = 256
params['max_decode_length'] = 4
```

## What is a Cache?
In auto-regressive architectures like Transformer based [Encoder-Decoder](https://arxiv.org/abs/1706.03762) models, 
Cache is used for fast sequential decoding.
It is a nested dictionary storing pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) for every layer.

```
{
    'layer_%d' % layer: {
        'k': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'], params['n_dims']/params['num_heads']], dtype=tf.float32),
        'v': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'], params['n_dims']/params['num_heads']], dtype=tf.float32)
        } for layer in range(params['num_layers']),
    'model_specific_item' : Model specific tensor shape,
}

```

# Initialize cache. 


```python
cache = {
    'layer_%d' % layer: {
        'k': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'], params['n_dims']/params['num_heads']], dtype=tf.float32),
        'v': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'], params['n_dims']/params['num_heads']], dtype=tf.float32)
        } for layer in range(params['num_layers'])
    }
print("cache key shape for layer 1 :", cache['layer_1']['k'].shape)
```

# Define closure for length normalization. **optional.**





```python
def length_norm(length, dtype):
  """Return length normalization factor."""
  return tf.pow(((5. + tf.cast(length, dtype)) / 6.), 0.0)
```

# Create model_fn
  In practice, this will be replaced by an actual model implementation such as [here](https://github.com/tensorflow/models/blob/master/official/nlp/transformer/transformer.py#L236)
```
Args:
i : Step that is being decoded.
Returns:
  logit probabilities of size [batch_size, 1, vocab_size]
```




```python
probabilities = tf.constant([[[0.3, 0.4, 0.3], [0.3, 0.3, 0.4],
                              [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                            [[0.2, 0.5, 0.3], [0.2, 0.7, 0.1],
                              [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]])
def model_fn(i):
  return probabilities[:, i, :]
```

# Initialize symbols_to_logits_fn



```python
def _symbols_to_logits_fn():
  """Calculates logits of the next tokens."""
  def symbols_to_logits_fn(ids, i, temp_cache):
    del ids
    logits = tf.cast(tf.math.log(model_fn(i)), tf.float32)
    return logits, temp_cache
  return symbols_to_logits_fn
```

# Greedy 
Greedy decoding selects the token id with the highest probability as its next id: $id_t = argmax_{w}P(id | id_{1:t-1})$ at each timestep $t$. The following sketch shows greedy decoding. 


```python
greedy_obj = sampling_module.SamplingModule(
    length_normalization_fn=None,
    dtype=tf.float32,
    symbols_to_logits_fn=_symbols_to_logits_fn(),
    vocab_size=3,
    max_decode_length=params['max_decode_length'],
    eos_id=10,
    padded_decode=False)
ids, _ = greedy_obj.generate(
    initial_ids=tf.constant([9, 1]), initial_cache=cache)
print("Greedy Decoded Ids:", ids)
```

# top_k sampling
In *Top-K* sampling, the *K* most likely next token ids are filtered and the probability mass is redistributed among only those *K* ids. 


```python
top_k_obj = sampling_module.SamplingModule(
    length_normalization_fn=length_norm,
    dtype=tf.float32,
    symbols_to_logits_fn=_symbols_to_logits_fn(),
    vocab_size=3,
    max_decode_length=params['max_decode_length'],
    eos_id=10,
    sample_temperature=tf.constant(1.0),
    top_k=tf.constant(3),
    padded_decode=False,
    enable_greedy=False)
ids, _ = top_k_obj.generate(
    initial_ids=tf.constant([9, 1]), initial_cache=cache)
print("top-k sampled Ids:", ids)
```

# top_p sampling
Instead of sampling only from the most likely *K* token ids, in *Top-p* sampling chooses from the smallest possible set of ids whose cumulative probability exceeds the probability *p*.


```python
top_p_obj = sampling_module.SamplingModule(
    length_normalization_fn=length_norm,
    dtype=tf.float32,
    symbols_to_logits_fn=_symbols_to_logits_fn(),
    vocab_size=3,
    max_decode_length=params['max_decode_length'],
    eos_id=10,
    sample_temperature=tf.constant(1.0),
    top_p=tf.constant(0.9),
    padded_decode=False,
    enable_greedy=False)
ids, _ = top_p_obj.generate(
    initial_ids=tf.constant([9, 1]), initial_cache=cache)
print("top-p sampled Ids:", ids)
```

# Beam search decoding
Beam search reduces the risk of missing hidden high probability token ids by keeping the most likely num_beams of hypotheses at each time step and eventually choosing the hypothesis that has the overall highest probability. 


```python
beam_size = 2
params['batch_size'] = 1
beam_cache = {
    'layer_%d' % layer: {
        'k': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'], params['n_dims']], dtype=tf.float32),
        'v': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'], params['n_dims']], dtype=tf.float32)
        } for layer in range(params['num_layers'])
    }
print("cache key shape for layer 1 :", beam_cache['layer_1']['k'].shape)
ids, _ = beam_search.sequence_beam_search(
    symbols_to_logits_fn=_symbols_to_logits_fn(),
    initial_ids=tf.constant([9], tf.int32),
    initial_cache=beam_cache,
    vocab_size=3,
    beam_size=beam_size,
    alpha=0.6,
    max_decode_length=params['max_decode_length'],
    eos_id=10,
    padded_decode=False,
    dtype=tf.float32)
print("Beam search ids:", ids)
```
