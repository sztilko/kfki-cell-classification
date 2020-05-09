from model import build_and_train_rnn
from utils import print_json, save_json_result, load_best_hyperspace,print_model_data
from hyperopt import hp, tpe, fmin, Trials
import tensorflow as tf
import pickle
import os


space = {
    'hidden_dim': hp.quniform('hidden_dim', 32, 256,1),
    'batch_size': hp.quniform('batch_size', 100, 450, 5),
    'cell_type': hp.choice('cell_type', ['LSTM', 'GRU']),
    'normalize_data' : hp.choice('normalize_data', [True, False]),
    'depth' : hp.choice('depth', [1, 2]),
    'fc_dim' : hp.choice('fc_status',[None, hp.quniform('fc_dim', 2, 64, 1)])
}


def print_best_model():
    space_best_model = load_best_hyperspace()
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    print_model_data(['model_name','train_accuracy','test_accuracy','model_param_num'])
    print_json(space_best_model)


def optimize_rnn(hype_space):
    model, model_name, result = build_and_train_rnn(hype_space)

    # Save training results to disks with unique filenames
    save_json_result(model_name, result)

    tf.keras.backend.clear_session()
    del model
    print("\n\n")
    
    return result

def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        optimize_rnn,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")


if __name__ == "__main__":
    while True:

      # Optimize a new model with the TPE Algorithm:
      print("OPTIMIZING NEW MODEL:")
      run_a_trial()

      # Replot best model since it may have changed:
      print("PLOTTING YET BEST MODEL:")
      print_best_model()
