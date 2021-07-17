##### Copyright 2020 The TensorFlow Authors.


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

# Customizing a Transformer Encoder

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/official_models/nlp/customize_encoder"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/models/blob/master/official/colab/nlp/customize_encoder.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/models/blob/master/official/colab/nlp/customize_encoder.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/models/official/colab/nlp/customize_encoder.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

## Learning objectives

The [TensorFlow Models NLP library](https://github.com/tensorflow/models/tree/master/official/nlp/modeling) is a collection of tools for building and training modern high performance natural language models.

The [TransformEncoder](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/encoder_scaffold.py) is the core of this library, and lots of new network architectures are proposed to improve the encoder. In this Colab notebook, we will learn how to customize the encoder to employ new network architectures.

## Install and import

### Install the TensorFlow Model Garden pip package

*  `tf-models-official` is the stable Model Garden package. Note that it may not include the latest changes in the `tensorflow_models` github repo. To include latest changes, you may install `tf-models-nightly`,
which is the nightly Model Garden package created daily automatically.
*  `pip` will install all models and dependencies automatically.


```python
!pip install -q tf-models-official==2.3.0
```

### Import Tensorflow and other libraries


```python
import numpy as np
import tensorflow as tf

from official.modeling import activations
from official.nlp import modeling
from official.nlp.modeling import layers, losses, models, networks
```

## Canonical BERT encoder

Before learning how to customize the encoder, let's firstly create a canonical BERT enoder and use it to instantiate a `BertClassifier` for classification task.


```python
cfg = {
    "vocab_size": 100,
    "hidden_size": 32,
    "num_layers": 3,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "activation": activations.gelu,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "sequence_length": 16,
    "type_vocab_size": 2,
    "initializer": tf.keras.initializers.TruncatedNormal(stddev=0.02),
}
bert_encoder = modeling.networks.TransformerEncoder(**cfg)

def build_classifier(bert_encoder):
  return modeling.models.BertClassifier(bert_encoder, num_classes=2)

canonical_classifier_model = build_classifier(bert_encoder)
```

`canonical_classifier_model` can be trained using the training data. For details about how to train the model, please see the colab [fine_tuning_bert.ipynb](https://github.com/tensorflow/models/blob/master/official/colab/fine_tuning_bert.ipynb). We skip the code that trains the model here.

After training, we can apply the model to do prediction.



```python
def predict(model):
  batch_size = 3
  np.random.seed(0)
  word_ids = np.random.randint(
      cfg["vocab_size"], size=(batch_size, cfg["sequence_length"]))
  mask = np.random.randint(2, size=(batch_size, cfg["sequence_length"]))
  type_ids = np.random.randint(
      cfg["type_vocab_size"], size=(batch_size, cfg["sequence_length"]))
  print(model([word_ids, mask, type_ids], training=False))

predict(canonical_classifier_model)
```

## Customize BERT encoder

One BERT encoder consists of an embedding network and multiple transformer blocks, and each transformer block contains an attention layer and a feedforward layer.

We provide easy ways to customize each of those components via (1)
[EncoderScaffold](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/encoder_scaffold.py) and (2) [TransformerScaffold](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer_scaffold.py).

### Use EncoderScaffold

`EncoderScaffold` allows users to provide a custom embedding subnetwork
  (which will replace the standard embedding logic) and/or a custom hidden layer class (which will replace the `Transformer` instantiation in the encoder).

#### Without Customization

Without any customization, `EncoderScaffold` behaves the same the canonical `TransformerEncoder`.

As shown in the following example, `EncoderScaffold` can load `TransformerEncoder`'s weights and output the same values:


```python
default_hidden_cfg = dict(
    num_attention_heads=cfg["num_attention_heads"],
    intermediate_size=cfg["intermediate_size"],
    intermediate_activation=activations.gelu,
    dropout_rate=cfg["dropout_rate"],
    attention_dropout_rate=cfg["attention_dropout_rate"],
    kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
)
default_embedding_cfg = dict(
    vocab_size=cfg["vocab_size"],
    type_vocab_size=cfg["type_vocab_size"],
    hidden_size=cfg["hidden_size"],
    seq_length=cfg["sequence_length"],
    initializer=tf.keras.initializers.TruncatedNormal(0.02),
    dropout_rate=cfg["dropout_rate"],
    max_seq_length=cfg["sequence_length"],
)
default_kwargs = dict(
    hidden_cfg=default_hidden_cfg,
    embedding_cfg=default_embedding_cfg,
    num_hidden_instances=cfg["num_layers"],
    pooled_output_dim=cfg["hidden_size"],
    return_all_layer_outputs=True,
    pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(0.02),
)
encoder_scaffold = modeling.networks.EncoderScaffold(**default_kwargs)
classifier_model_from_encoder_scaffold = build_classifier(encoder_scaffold)
classifier_model_from_encoder_scaffold.set_weights(
    canonical_classifier_model.get_weights())
predict(classifier_model_from_encoder_scaffold)
```

#### Customize Embedding

Next, we show how to use a customized embedding network.

We firstly build an embedding network that will replace the default network. This one will have 2 inputs (`mask` and `word_ids`) instead of 3, and won't use positional embeddings.


```python
word_ids = tf.keras.layers.Input(
    shape=(cfg['sequence_length'],), dtype=tf.int32, name="input_word_ids")
mask = tf.keras.layers.Input(
    shape=(cfg['sequence_length'],), dtype=tf.int32, name="input_mask")
embedding_layer = modeling.layers.OnDeviceEmbedding(
    vocab_size=cfg['vocab_size'],
    embedding_width=cfg['hidden_size'],
    initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
    name="word_embeddings")
word_embeddings = embedding_layer(word_ids)
attention_mask = layers.SelfAttentionMask()([word_embeddings, mask])
new_embedding_network = tf.keras.Model([word_ids, mask],
                                       [word_embeddings, attention_mask])
```

Inspecting `new_embedding_network`, we can see it takes two inputs:
`input_word_ids` and `input_mask`.


```python
tf.keras.utils.plot_model(new_embedding_network, show_shapes=True, dpi=48)
```

We then can build a new encoder using the above `new_embedding_network`.


```python
kwargs = dict(default_kwargs)

# Use new embedding network.
kwargs['embedding_cls'] = new_embedding_network
kwargs['embedding_data'] = embedding_layer.embeddings

encoder_with_customized_embedding = modeling.networks.EncoderScaffold(**kwargs)
classifier_model = build_classifier(encoder_with_customized_embedding)
# ... Train the model ...
print(classifier_model.inputs)

# Assert that there are only two inputs.
assert len(classifier_model.inputs) == 2
```

#### Customized Transformer

User can also override the [hidden_cls](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/encoder_scaffold.py#L103) argument in `EncoderScaffold`'s constructor to employ a customized Transformer layer.

See [ReZeroTransformer](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/rezero_transformer.py) for how to implement a customized Transformer layer.

Following is an example of using `ReZeroTransformer`:



```python
kwargs = dict(default_kwargs)

# Use ReZeroTransformer.
kwargs['hidden_cls'] = modeling.layers.ReZeroTransformer

encoder_with_rezero_transformer = modeling.networks.EncoderScaffold(**kwargs)
classifier_model = build_classifier(encoder_with_rezero_transformer)
# ... Train the model ...
predict(classifier_model)

# Assert that the variable `rezero_alpha` from ReZeroTransformer exists.
assert 'rezero_alpha' in ''.join([x.name for x in classifier_model.trainable_weights])
```

### Use [TransformerScaffold](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer_scaffold.py)

The above method of customizing `Transformer` requires rewriting the whole `Transformer` layer, while sometimes you may only want to customize either attention layer or feedforward block. In this case, [TransformerScaffold](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer_scaffold.py) can be used.



#### Customize Attention Layer

User can also override the [attention_cls](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/transformer_scaffold.py#L45) argument in `TransformerScaffold`'s constructor to employ a customized Attention layer.

See [TalkingHeadsAttention](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/talking_heads_attention.py) for how to implement a customized `Attention` layer.

Following is an example of using [TalkingHeadsAttention](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/talking_heads_attention.py):


```python
# Use TalkingHeadsAttention
hidden_cfg = dict(default_hidden_cfg)
hidden_cfg['attention_cls'] = modeling.layers.TalkingHeadsAttention

kwargs = dict(default_kwargs)
kwargs['hidden_cls'] = modeling.layers.TransformerScaffold
kwargs['hidden_cfg'] = hidden_cfg

encoder = modeling.networks.EncoderScaffold(**kwargs)
classifier_model = build_classifier(encoder)
# ... Train the model ...
predict(classifier_model)

# Assert that the variable `pre_softmax_weight` from TalkingHeadsAttention exists.
assert 'pre_softmax_weight' in ''.join([x.name for x in classifier_model.trainable_weights])
```

#### Customize Feedforward Layer

Similiarly, one could also customize the feedforward layer.

See [GatedFeedforward](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/gated_feedforward.py) for how to implement a customized feedforward layer.

Following is an example of using [GatedFeedforward](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/gated_feedforward.py).


```python
# Use TalkingHeadsAttention
hidden_cfg = dict(default_hidden_cfg)
hidden_cfg['feedforward_cls'] = modeling.layers.GatedFeedforward

kwargs = dict(default_kwargs)
kwargs['hidden_cls'] = modeling.layers.TransformerScaffold
kwargs['hidden_cfg'] = hidden_cfg

encoder_with_gated_feedforward = modeling.networks.EncoderScaffold(**kwargs)
classifier_model = build_classifier(encoder_with_gated_feedforward)
# ... Train the model ...
predict(classifier_model)

# Assert that the variable `gate` from GatedFeedforward exists.
assert 'gate' in ''.join([x.name for x in classifier_model.trainable_weights])
```

### Build a new Encoder using building blocks from KerasBERT.

Finally, you could also build a new encoder using building blocks in the modeling library.

See [AlbertTransformerEncoder](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/albert_transformer_encoder.py) as an example:



```python
albert_encoder = modeling.networks.AlbertTransformerEncoder(**cfg)
classifier_model = build_classifier(albert_encoder)
# ... Train the model ...
predict(classifier_model)
```

Inspecting the `albert_encoder`, we see it stacks the same `Transformer` layer multiple times.


```python
tf.keras.utils.plot_model(albert_encoder, show_shapes=True, dpi=48)
```
