import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pprint
import os
import re
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Metric, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def get_lightpaths(service_file, graph) -> list:
    """
    Function taken from Soham's RL model. Given a gnpy output file of a
    topology snapshot, get a list of candidate paths (not trails or cycles).
    These are stored as a list of dictionaries containing gnpy simulation
    data.
    :param service_file: file path to gnpy output file
    :return retval: list of dictionaries containing path data [{p1}{p2}{p3}]
    """
    node_set = set(graph.nodes)
    with open(service_file, 'r') as file:
        # Load the JSON data into a Python dictionary
        responses = json.load(file)['response']
        # pprint.pprint(responses)
        metrics = set(["SNR-bandwidth", "SNR-0.1nm", "OSNR-bandwidth", "OSNR-0.1nm"])

    retval = []
    for i in responses:
        if "path-properties" not in i:
            continue
        path_route_objects = i["path-properties"]["path-route-objects"]
        path_metric = i["path-properties"]["path-metric"]
        curr_path = []
        for j in path_route_objects:
            if "num-unnum-hop" in j["path-route-object"]:
                curr_node = j["path-route-object"]["num-unnum-hop"]["node-id"]
                if curr_node in node_set:
                    curr_path.append(curr_node)
        if len(curr_path) > 0:
            my_dict = {}
            my_dict["path"] = curr_path
            for m in path_metric:
                if m["metric-type"] in metrics:
                    my_dict[m["metric-type"]] = m["accumulative-value"]
            retval.append(my_dict)

    return retval


# F1 score
class F1Score(Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = Precision(thresholds=self.threshold)
        self.recall = Recall(thresholds=self.threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


if __name__ == "__main__":
    # networkx graph gmls
    top1_phys = nx.read_gml("topology files/top1_phys.gml") # imported all of them but only 1 is used.
    top1_full = nx.read_gml("topology files/top1_full.gml")
    top1_star = nx.read_gml("topology files/top1_star.gml")

    top2_phys = nx.read_gml("topology files/top2_phys.gml")
    top2_full = nx.read_gml("topology files/top2_full.gml")
    top2_star = nx.read_gml("topology files/top2_star.gml")

    top3_phys = nx.read_gml("topology files/top3_phys.gml")
    top3_star = nx.read_gml("topology files/top3_star.gml")


    # folder names storing output files of gnpy simulations

    # old experiments, static failure
    # folder_i = "0.3-200"
    # folder_i = "0.4-200"
    # folder_i = "0.45-200"
    # folder_i = "0.5-500"
    # folder_i = "0.6-200"
    # folder_i = "0.65-200"
    # folder_i = "0.66-200"
    # folder_i = "0.68-200"
    # folder_i = "0.7-1000"

    # new experiments, standard deviation failure
    folder_i =  "0.2,0.5-200"

    """
    folders = [ f'soft failure/gnpy soft failure data/{folder_i}/fiber (dA_v1 → dC_v1)_(2 of 2)',
                f'soft failure/gnpy soft failure data/{folder_i}/fiber (dB_v4 → dD_v1)_(2 of 2)',
                f'soft failure/gnpy soft failure data/{folder_i}/fiber (dD_v2 → dC_v2)_(2 of 2)' ]
    """
    folders = [ f'soft failure/gnpy soft failure data/{folder_i}/fiber (dA_v1 → dC_v1)_(2 of 2)',
                f'soft failure/gnpy soft failure data/{folder_i}/fiber (dB_v4 → dD_v1)_(2 of 2)',
                f'soft failure/gnpy soft failure data/{folder_i}/fiber (dD_v2 → dC_v2)_(2 of 2)',
                f'soft failure/gnpy soft failure data/{folder_i}/regular_traffic']
    #"""

    # regex pattern to extract timestep number from filename, e.g., "output_file_12.json"
    pattern = re.compile(r'output_file_(\d+)\.json')

    # collect all files with their source folder and timestep
    all_files = []

    for folder in folders:
        for filename in os.listdir(folder):
            match = pattern.match(filename)
            if match:
                timestep = int(match.group(1))
                filepath = os.path.join(folder, filename)
                all_files.append((timestep, filepath, folder))

    # Sort the list by timestep
    sorted_files = sorted(all_files, key=lambda x: x[0])

    X = []
    Y = []
    for timestep, path, folder in sorted_files:
        data = get_lightpaths(path, top2_full)
        ts = [] # data at each timestep for each path
        for item in data:
            # pprint.pprint(item)
            flattened_item = [
                item['OSNR-0.1nm'],
                item['OSNR-bandwidth'],
                item['SNR-0.1nm'],
                item['SNR-bandwidth'],
            ]
            ts.append(flattened_item)
        X.append(ts)

        print(folder)
        match folder:
            case a if a == f'soft failure/gnpy soft failure data/{folder_i}/fiber (dA_v1 → dC_v1)_(2 of 2)':
                Y.append([1, 1, 0, 0])

            case b if b == f'soft failure/gnpy soft failure data/{folder_i}/fiber (dB_v4 → dD_v1)_(2 of 2)':
                Y.append([0, 0, 1, 0])

            case c if c ==f'soft failure/gnpy soft failure data/{folder_i}/fiber (dD_v2 → dC_v2)_(2 of 2)':
                Y.append([0, 1, 1, 1])

            case d if d ==f'soft failure/gnpy soft failure data/{folder_i}/regular_traffic':
                Y.append([0, 0, 0, 0])

    X = np.array(X)
    pprint.pprint(X)
    Y = np.array(Y)
    pprint.pprint(Y)

    """
    # VOILIN Plots
    # Flatten and organize into a DataFrame
    records = []
    trail_names = ['Trail 1', 'Trail 2', 'Trail 3', 'Trail 4']
    feature_names = ['OSNR-0.1nm', 'OSNR-Band', 'SNR-0.1nm', 'SNR-Band']

    for t in range(X.shape[0]):
        for trail in range(4):
            for f_idx, feature in enumerate(feature_names):
                records.append({
                    'Trail': trail_names[trail],
                    'Feature': feature,
                    'Value': X[t, trail, f_idx],
                    'Failure': Y[t, trail]  # 0 or 1
                })

    # Create DataFrame
    df = pd.DataFrame(records)

    # Plot
    g = sns.catplot(
        data=df, kind="violin",
        x="Feature", y="Value", hue="Failure",
        col="Trail", split=True,
        height=4, aspect=1.2, sharey=False
    )

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Violin Plots of X Features by Failure Status per Monitoring Trail")
    plt.show()
    """

    # train test split
    split_ratio = 0.8
    split_idx = int(split_ratio * len(X))
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]

    # reshape
    time_steps_train = X_train.shape[0]
    time_steps_test = X_test.shape[0]
    num_trails = X.shape[1]
    num_features = X.shape[2]
    output_size = Y.shape[1]

    X_train_model = X_train.reshape(1, time_steps_train, num_trails * num_features)
    Y_train_model = Y_train.reshape(1, time_steps_train, output_size)
    X_test_model = X_test.reshape(1, time_steps_test, num_trails * num_features)
    Y_test_model = Y_test.reshape(1, time_steps_test, output_size)

    # define model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(None, num_trails * num_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        TimeDistributed(Dense(output_size, activation='sigmoid'))
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            BinaryAccuracy(name='binary_accuracy', threshold=0.5),
            Precision(),
            Recall(),
            AUC(),
            F1Score()
        ]
    )

    # early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=100,
        restore_best_weights=True
    )

    # training
    history = model.fit(
        X_train_model,
        Y_train_model,
        epochs=1000,
        batch_size=1,
        validation_data=(X_test_model, Y_test_model),
        callbacks=[early_stop],     # [early_stop] for early stopping
        verbose=1
    )

    # evaluate
    loss, acc, prec, rec, auc, f1 = model.evaluate(X_test_model, Y_test_model)
    print(f"Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f}  | F1: {f1:.4f}")

    # plot data
    metrics = ['loss', 'binary_accuracy', 'precision', 'recall', 'auc', 'f1_score']
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()
    plt.show()

    # manually print accuracy (to check)
    Y_pred_probs = model.predict(X_test_model)
    Y_pred_binary = (Y_pred_probs > 0.5).astype(int)

    # accuracy (manually)
    correct_total = 0
    total_values = 0

    for i in range(Y_test_model.shape[1]):  # loop over time steps
        probs   = Y_pred_probs[0][i]
        binary  = Y_pred_binary[0][i]
        actual  = Y_test_model[0][i]

        # compare prediction to actual
        matches = (binary == actual).astype(int)
        correct_total += np.sum(matches)
        total_values += len(matches)

        print(f"Time Step {i+1}:")
        print("  Predicted Probabilities:", [f"{p:.4f}" for p in probs])
        print("  Predicted Labels:       ", binary.tolist())
        print("  Actual Labels:          ", actual.tolist())
        print("  Correct Predictions:    ", matches.tolist())
        print()

        # final summary
        accuracy = correct_total / total_values if total_values > 0 else 0
        print(f"Total Correct Predictions: {correct_total}/{total_values}")
        print(f"Manual Accuracy: {accuracy:.4f}")
