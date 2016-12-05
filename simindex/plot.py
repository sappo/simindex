from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import (precision_recall_curve,
                             recall_score,
                             precision_score)

# setup plot details
lw = 2


def show():
    plt.show()


def draw_frequency_distribution(data, dataset, type):
    print("Data:", data)
    fig = plt.figure(dpi=None, facecolor="white")
    fig.canvas.set_window_title("Frequency Distribution (%s)" % dataset)
    plt.clf()
    # max = 0
    for index, label in enumerate(data.keys()):
        x_vals, y_vals = [], []
        for y, x in sorted(data[label].items(), key=lambda x: (x[1], -int(x[0]))):
            x_vals.append(x)
            y_vals.append(int(y))

        # max = sum([x * y for x, y in zip(x_vals, y_vals)])
        plt.plot(x_vals, y_vals, "o-", lw=lw, label=label)

    # zipf_data = Counter(np.random.zipf(3., max))
    # x_zipfs, y_zipfs = [], []
    # for y, x in sorted(zipf_data.items(), key=lambda x: (x[1], -x[0])):
        # x_zipfs.append(x)
        # y_zipfs.append(y)

    # plt.plot(x_zipfs, y_zipfs, "--", lw=lw, label="Zipfian")

    plt.xlabel("%ss sorted by number of items per %s" % (type, type))
    plt.ylabel("Number of items per %s" % type)
    plt.autoscale(enable=True, axis='x')
    plt.autoscale(enable=True, axis='y')
    plt.title("Frequency of %s" % type)
    plt.legend(loc="upper right")


def draw_record_time_curve(times, dataset, action):
    title = "%s time for a single record (%s)" % (action.capitalize(), dataset)
    fig = plt.figure(dpi=None, facecolor="white")
    fig.canvas.set_window_title(title)
    index = np.arange(0, len(times), 1)
    mean = np.mean(times)
    times_mean = [mean for i in index]
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(index, times, lw=lw, color='navy', label='Record-Time')
    plt.plot(index, times_mean, color='red', label='Mean', linestyle='--')
    xlabel = "Record %s number" % action
    plt.xlabel(xlabel)
    plt.ylabel('Time (s)')
    plt.ylim([pow(10, -5), pow(10, -1)])
    plt.autoscale(enable=True, axis='y')
    plt.xlim([0, max(index)])
    plt.title(title)
    plt.legend(loc="upper right")


def draw_precision_recall_curve(y_true, y_scores, dataset):
    fig = plt.figure(dpi=None, facecolor="white")
    fig.canvas.set_window_title("PRC (%s)" % dataset)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if (thresholds[0] == 0):
        recall = recall[1:]
        precision = precision[1:]
    print(thresholds)
    print(recall)
    print(precision)
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, 'yo-', lw=lw, color='navy', picker=True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    title = "Precision-Recall curve (%s)" % dataset
    plt.title(title)
    plt.legend(loc="lower left")
