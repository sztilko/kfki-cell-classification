from utils import print_json,return_dataset

from tensorflow.keras.layers import InputLayer, Dense,LSTM,GRU
from tensorflow.keras.models import Sequential
import tensorflow as tf
from hyperopt import STATUS_OK, STATUS_FAIL


import uuid
import traceback
import os

# You may want to reduce this considerably if you don't have a killer GPU:
EPOCHS = 150
TENSORBOARD_DIR = "TensorBoard/"


train_data_path = 'gs://kfki-single-cell-fibronectin/merged_corrected_data/train.csv'
test_data_path = 'gs://kfki-single-cell-fibronectin/merged_corrected_data/test.csv'


def build_and_train_rnn(hype_space,log_for_tensorboard=True):

    train_x, train_y = return_dataset(train_data_path,hype_space['normalize_data'])
    test_x, test_y = return_dataset(test_data_path,hype_space['normalize_data'])

    model = Sequential()
    model.add(InputLayer(input_shape=(600,1)))
    if hype_space['depth']==1:
      model.add(return_cell_type(hype_space['cell_type'])(int(hype_space['hidden_dim'])))
    if hype_space['depth']==2:
      model.add(return_cell_type(hype_space['cell_type'])(int(hype_space['hidden_dim']),return_sequences=True))
      model.add(return_cell_type(hype_space['cell_type'])(int(hype_space['hidden_dim'])))
    if hype_space['fc_dim']:
      model.add(Dense(int(hype_space['fc_dim']), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_uuid = str(uuid.uuid4())[:5]

    # TensorBoard logging callback:
    log_path = None
    if log_for_tensorboard:
        log_path = os.path.join(TENSORBOARD_DIR, model_uuid)
        print("Tensorboard log files will be saved to: {}".format(log_path))
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_path,
            write_graph=True
        )
        tb_callback.set_model(model)
    

    # Train net:
    history = model.fit(
        train_x,
        train_y,
        batch_size=int(hype_space['batch_size']),
        epochs=EPOCHS,
        shuffle=True,
        verbose=0,
        callbacks = [tb_callback]
    ).history

    # Test net:
    print(f'History keys: {history.keys()}')
    score = model.evaluate(test_x, test_y, verbose=0)

    model_name = "model_{:.4f}_{}".format(score[1], model_uuid)
    print("Model name: {}".format(model_name))

    result = {
        'train_accuracy': history['accuracy'][-1],
        'test_accuracy': score[1],
        'loss' : score[0], #minimize test loss as hypertuning objective
        # Misc:
        'model_name': model_name,
        'space': hype_space,
        'history': history,
        'status': STATUS_OK,
        'model_param_num' : model.count_params()
    }

    # print("RESULT:")
    # print_json(result)

    return model, model_name, result

def return_cell_type(cell_type):
  if cell_type=='LSTM':
    return LSTM
  if cell_type=='GRU':
    return GRU
