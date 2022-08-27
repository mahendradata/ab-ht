from eil.dataset import BatchData
from eil.util import Log, Timer, counter, report, confusion_matrix


def evaluate_model(name: str, model, dataset_paths: str, samples: BatchData, random_state=10):

    batch = 0  # batch id
    for f in dataset_paths:
        batch += 1  # batch id is increased for each dataset
        Log.write("Reading:", f)

        # Split to train and test dataset
        train_x, eval_x, train_y, eval_y = samples.read(f, random_state=random_state)

        # Train
        timer = Timer()
        model.fit(train_x, train_y)
        Log.write("{}; training duration; {}; {}".format(name, counter(train_y), timer.stop()))

        # Evaluate
        timer = Timer()
        y_pred = model.predict(eval_x)
        duration = timer.stop()
        train_y_count = counter(train_y)
        Log.write("{}; evaluation duration; {}; {}".format(name, train_y_count, duration))

        # Make report
        report_str, report_dict = report(eval_y, y_pred)  # Generate classification report
        Log.write_exp("{}.{}".format(name, batch), "eval", report_dict)
        Log.write(report_str)

        # Make confusion matrix
        matrix = confusion_matrix(eval_y, y_pred)
        Log.write("{}.{}; eval matrix; {}; Predicted:row / Actual:col;\n{}".format(name, batch, train_y_count, matrix))
        # noinspection PyTypeChecker
        matrix.to_csv("{}.{}.csv".format(Log.BASE_NAME, batch), index=True)
