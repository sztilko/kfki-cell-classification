from bson import json_util
import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import tensorflow as tf

RESULTS_DIR = "results/"


def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join(RESULTS_DIR, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
            # default=json_util.default,
            # separators=(',', ': ')
        )

def load_best_hyperspace():
    results = [
        f for f in list(sorted(os.listdir(RESULTS_DIR))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]

def return_dataset(path):
  data = pd.read_csv(path)
  data.pop(data.columns[0]) #pop the original index
  data_x = data[data.columns[1:]]
  # "normalize" data
  # data_x=data_x/1500
  data_x = np.expand_dims(np.asarray(data_x).astype('float32'),-1)

  data_y = data['0'] # 'hela' or 'preo' string labels
  label_encoder = LabelEncoder()
  label_int = label_encoder.fit_transform(data_y) # convert to int labels
  data_y = tf.keras.utils.to_categorical(label_int,num_classes=2) # convert to one-hot labels
 
  return data_x, data_y
