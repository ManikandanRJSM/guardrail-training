# !pip install -U "tensorflow-text==2.18.*"
# !pip install sklearn
# !pip install pandas
# !pip install tf-models-official

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import tf_keras

import matplotlib.pyplot as plt

print("TF Version:", tf.__version__)
print("Keras Version:", tf_keras.__version__)

tf.get_logger().setLevel('ERROR')





AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42



from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('./data_lake/guardrails_inputs/guardrails_inputs.csv')
texts = df['text'].values
labels = df['label'].values

# First split → separate test set (15%)
train_val_texts, test_texts, \
train_val_labels, test_labels = train_test_split(
    texts, labels,
    test_size=0.15,
    random_state=42)

# Second split → separate val from train (15%)
train_texts, val_texts, \
train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels,
    test_size=0.15,
    random_state=42)


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_texts, train_labels)) \
    .batch(32) \
    .cache() \
    .prefetch(AUTOTUNE)

# Extract unique class names from the labels
class_names = np.unique(labels).tolist()

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_texts, val_labels)) \
    .batch(32) \
    .cache() \
    .prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_texts, test_labels)) \
    .batch(32) \
    .cache() \
    .prefetch(AUTOTUNE)



tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')


bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')


bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_results = bert_model(text_preprocessed)
print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)



classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

# tf.keras.utils.plot_model(classifier_model)
# compile the model before start the training:
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')


classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)



print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)