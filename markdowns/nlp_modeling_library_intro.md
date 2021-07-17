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

# Introduction to the TensorFlow Models NLP library

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/official_models/nlp/nlp_modeling_library_intro"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/models/blob/master/official/colab/nlp/nlp_modeling_library_intro.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/models/blob/master/official/colab/nlp/nlp_modeling_library_intro.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/models/official/colab/nlp/nlp_modeling_library_intro.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

## Learning objectives

In this Colab notebook, you will learn how to build transformer-based models for common NLP tasks including pretraining, span labelling and classification using the building blocks from [NLP modeling library](https://github.com/tensorflow/models/tree/master/official/nlp/modeling).

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

from official.nlp import modeling
from official.nlp.modeling import layers, losses, models, networks
```

## BERT pretraining model

BERT ([Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)) introduced the method of pre-training language representations on a large text corpus and then using that model for downstream NLP tasks.

In this section, we will learn how to build a model to pretrain BERT on the masked language modeling task and next sentence prediction task. For simplicity, we only show the minimum example and use dummy data.

### Build a `BertPretrainer` model wrapping `TransformerEncoder`

The [TransformerEncoder](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/networks/transformer_encoder.py) implements the Transformer-based encoder as described in [BERT paper](https://arxiv.org/abs/1810.04805). It includes the embedding lookups and transformer layers, but not the masked language model or classification task networks.

The [BertPretrainer](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/bert_pretrainer.py) allows a user to pass in a transformer stack, and instantiates the masked language model and classification networks that are used to create the training objectives.


```python
# Build a small transformer network.
vocab_size = 100
sequence_length = 16
network = modeling.networks.TransformerEncoder(
    vocab_size=vocab_size, num_layers=2, sequence_length=16)
```

Inspecting the encoder, we see it contains few embedding layers, stacked `Transformer` layers and are connected to three input layers:

`input_word_ids`, `input_type_ids` and `input_mask`.



```python
tf.keras.utils.plot_model(network, show_shapes=True, dpi=48)
```


```python
# Create a BERT pretrainer with the created network.
num_token_predictions = 8
bert_pretrainer = modeling.models.BertPretrainer(
    network, num_classes=2, num_token_predictions=num_token_predictions, output='predictions')
```

Inspecting the `bert_pretrainer`, we see it wraps the `encoder` with additional `MaskedLM` and `Classification` heads.


```python
tf.keras.utils.plot_model(bert_pretrainer, show_shapes=True, dpi=48)
```


```python
# We can feed some dummy data to get masked language model and sentence output.
batch_size = 2
word_id_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
mask_data = np.random.randint(2, size=(batch_size, sequence_length))
type_id_data = np.random.randint(2, size=(batch_size, sequence_length))
masked_lm_positions_data = np.random.randint(2, size=(batch_size, num_token_predictions))

outputs = bert_pretrainer(
    [word_id_data, mask_data, type_id_data, masked_lm_positions_data])
lm_output = outputs["masked_lm"]
sentence_output = outputs["classification"]
print(lm_output)
print(sentence_output)
```

### Compute loss
Next, we can use `lm_output` and `sentence_output` to compute `loss`.


```python
masked_lm_ids_data = np.random.randint(vocab_size, size=(batch_size, num_token_predictions))
masked_lm_weights_data = np.random.randint(2, size=(batch_size, num_token_predictions))
next_sentence_labels_data = np.random.randint(2, size=(batch_size))

mlm_loss = modeling.losses.weighted_sparse_categorical_crossentropy_loss(
    labels=masked_lm_ids_data,
    predictions=lm_output,
    weights=masked_lm_weights_data)
sentence_loss = modeling.losses.weighted_sparse_categorical_crossentropy_loss(
    labels=next_sentence_labels_data,
    predictions=sentence_output)
loss = mlm_loss + sentence_loss
print(loss)
```

With the loss, you can optimize the model.
After training, we can save the weights of TransformerEncoder for the downstream fine-tuning tasks. Please see [run_pretraining.py](https://github.com/tensorflow/models/blob/master/official/nlp/bert/run_pretraining.py) for the full example.



## Span labeling model

Span labeling is the task to assign labels to a span of the text, for example, label a span of text as the answer of a given question.

In this section, we will learn how to build a span labeling model. Again, we use dummy data for simplicity.

### Build a BertSpanLabeler wrapping TransformerEncoder

[BertSpanLabeler](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/bert_span_labeler.py) implements a simple single-span start-end predictor (that is, a model that predicts two values: a start token index and an end token index), suitable for SQuAD-style tasks.

Note that `BertSpanLabeler` wraps a `TransformerEncoder`, the weights of which can be restored from the above pretraining model.



```python
network = modeling.networks.TransformerEncoder(
        vocab_size=vocab_size, num_layers=2, sequence_length=sequence_length)

# Create a BERT trainer with the created network.
bert_span_labeler = modeling.models.BertSpanLabeler(network)
```

Inspecting the `bert_span_labeler`, we see it wraps the encoder with additional `SpanLabeling` that outputs `start_position` and `end_postion`.


```python
tf.keras.utils.plot_model(bert_span_labeler, show_shapes=True, dpi=48)
```


```python
# Create a set of 2-dimensional data tensors to feed into the model.
word_id_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
mask_data = np.random.randint(2, size=(batch_size, sequence_length))
type_id_data = np.random.randint(2, size=(batch_size, sequence_length))

# Feed the data to the model.
start_logits, end_logits = bert_span_labeler([word_id_data, mask_data, type_id_data])
print(start_logits)
print(end_logits)
```

### Compute loss
With `start_logits` and `end_logits`, we can compute loss:


```python
start_positions = np.random.randint(sequence_length, size=(batch_size))
end_positions = np.random.randint(sequence_length, size=(batch_size))

start_loss = tf.keras.losses.sparse_categorical_crossentropy(
    start_positions, start_logits, from_logits=True)
end_loss = tf.keras.losses.sparse_categorical_crossentropy(
    end_positions, end_logits, from_logits=True)

total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
print(total_loss)
```

With the `loss`, you can optimize the model. Please see [run_squad.py](https://github.com/tensorflow/models/blob/master/official/nlp/bert/run_squad.py) for the full example.

## Classification model

In the last section, we show how to build a text classification model.


### Build a BertClassifier model wrapping TransformerEncoder

[BertClassifier](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/bert_classifier.py) implements a [CLS] token classification model containing a single classification head.


```python
network = modeling.networks.TransformerEncoder(
        vocab_size=vocab_size, num_layers=2, sequence_length=sequence_length)

# Create a BERT trainer with the created network.
num_classes = 2
bert_classifier = modeling.models.BertClassifier(
    network, num_classes=num_classes)
```

Inspecting the `bert_classifier`, we see it wraps the `encoder` with additional `Classification` head.


```python
tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)
```


```python
# Create a set of 2-dimensional data tensors to feed into the model.
word_id_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
mask_data = np.random.randint(2, size=(batch_size, sequence_length))
type_id_data = np.random.randint(2, size=(batch_size, sequence_length))

# Feed the data to the model.
logits = bert_classifier([word_id_data, mask_data, type_id_data])
print(logits)
```

### Compute loss

With `logits`, we can compute `loss`:


```python
labels = np.random.randint(num_classes, size=(batch_size))

loss = modeling.losses.weighted_sparse_categorical_crossentropy_loss(
    labels=labels, predictions=tf.nn.log_softmax(logits, axis=-1))
print(loss)
```

With the `loss`, you can optimize the model. Please see [run_classifier.py](https://github.com/tensorflow/models/blob/master/official/nlp/bert/run_classifier.py) or the colab [fine_tuning_bert.ipynb](https://github.com/tensorflow/models/blob/master/official/colab/fine_tuning_bert.ipynb) for the full example.
