import numpy as np

from river import stream

from lib.dataset import IncrementalData
from lib.util import Log, Timer, counter, report, confusion_matrix


def evaluate_model(name: str, model, dataset_paths: str, samples: IncrementalData, random_state=10):

    batch = 0  # batch id
    for f in dataset_paths:
        batch += 1  # batch id is increased for each dataset
        Log.write("Reading:", f)

        # Split to train and test dataset
        train_x, eval_x, train_y, eval_y = samples.read(f, random_state=random_state)

        # Train
        timer = Timer()
        for x, y in stream.iter_pandas(train_x, train_y):
            model.learn_one(x, y)
        Log.write("{}; training duration; {}; {}".format(name, counter(train_y), timer.stop()))

        # Evaluate
        y_trues = list()
        y_preds = list()
        timer = Timer()
        for x, y in stream.iter_pandas(eval_x, eval_y):
            y_trues.append(y)
            y_preds.append(model.predict_one(x))  # Predict the evaluation data using the updated model
        duration = timer.stop()
        train_y_count = counter(train_y)
        Log.write("{}; evaluation duration; {}; {}".format(name, train_y_count, duration))

        # Make report
        y_trues = np.array(y_trues)
        y_preds = np.array(y_preds)
        report_str, report_dict = report(y_trues, y_preds)  # Generate classification report
        Log.write_exp("{}.{}".format(name, batch), "eval", report_dict)
        Log.write(report_str)

        # Make confusion matrix
        matrix = confusion_matrix(y_trues, y_preds)
        Log.write("{}.{}; eval matrix; {}; Predicted:row / Actual:col;\n{}".format(name, batch, train_y_count, matrix))
        # noinspection PyTypeChecker
        matrix.to_csv("{}.{}.csv".format(Log.BASE_NAME, batch), index=True)
