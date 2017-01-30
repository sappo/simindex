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

    # Sort data by value then label
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
    # Plot Precision-Recall curve
    plt.clf()
    # These are the colors that will be used in the plot
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    for index, indexer in enumerate(sorted(times.keys())):
        x_pos = np.arange(0, len(times[indexer]), 1)
        mean = np.mean(times[indexer])
        times_mean = [mean for i in x_pos]
        plt.plot(x_pos, times[indexer], lw=lw,
                 color=color_sequence[2 * index + 1], label=indexer)
        # zorder assures that mean lines are always printed above
        plt.plot(x_pos, times_mean, color=color_sequence[2 * index],
                 zorder=index + 100, linestyle='--', label='%s mean' % indexer)

    xlabel = "Record %s number" % action
    plt.xlabel(xlabel)
    plt.ylabel('Time (s)')
    plt.ylim([pow(10, -6), pow(10, 0)])
    plt.yscale('log')
    # plt.autoscale(enable=True, axis='y')
    plt.xlim([0, max(x_pos)])
    plt.title(title)
    plt.legend(loc="upper right")


def draw_precision_recall_curve(y_true, y_scores, dataset):
    fig = plt.figure(dpi=None, facecolor="white")
    fig.canvas.set_window_title("PRC (%s)" % dataset)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if (thresholds[0] == 0):
        recall = recall[1:]
        precision = precision[1:]
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


def draw_plots(x_vals, y_vals):
    fig = plt.figure(dpi=None, facecolor="white")
    fig.canvas.set_window_title("Plots")

    plt.clf()
    for y_val in y_vals:
        plt.plot(x_vals, y_val, 'yo-', lw=lw, color='navy', picker=True)

    fig.savefig("plot-%d.png" % len(x_vals))


def draw_bar_chart(data, title, unit):
    fig = plt.figure(dpi=None, facecolor="white")
    fig.canvas.set_window_title(title)
    plt.clf()
    plt.title(title)
    plt.ylabel(unit)

    def autolabel(rects):
        # Get y-axis height to calculate label position from.
        (y_bottom, y_top) = plt.ylim()
        y_height = y_top - y_bottom

        for rect in rects:
            height = rect.get_height()

            # Fraction of axis height taken up by this rectangle
            p_height = (height / y_height)

            # If we can fit the label above the column, do that;
            # otherwise, put it inside the column.
            if p_height > 0.95:  # arbitrary; 95% looked good to me.
                label_position = height - (y_height * 0.05)
            else:
                label_position = height + (y_height * 0.01)

            plt.text(rect.get_x() + rect.get_width() / 2., label_position,
                     str(round(height, 2)), ha='center', va='bottom')

    ax = plt.subplot(111)
    offset = 0.
    color_sequence = ['#1f77b4',  '#ff7f0e',  '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b',  '#e377c2',  '#7f7f7f',
                      '#bcbd22',  '#17becf']
    for index, indexer in enumerate(sorted(data.keys())):
        # Sort data by label
        y_vals = []
        objects = []
        for dataset, value in sorted(data[indexer].items(), key=lambda x: (x[0])):
            y_vals.append(value)
            objects.append(dataset)

        y_pos = [y + offset for y in np.arange(len(objects))]
        offset += 0.2
        bars = ax.bar(y_pos, y_vals, width=0.2, align="center",
                      alpha=0.8, color=color_sequence[index], label=indexer)
        autolabel(bars)

    y_pos = [y + offset / 4 for y in np.arange(len(objects))]
    plt.xticks(y_pos, objects)
    plt.legend(loc="best")
