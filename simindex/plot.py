import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from sklearn.metrics import precision_recall_curve

# setup plot details
lw = 2

plt.rcParams.update({'font.size': '12'})
plt.rcParams.update({'figure.titlesize': 'x-large'})
plt.rcParams.update({'axes.titlesize': 'x-large'})
plt.rcParams.update({'axes.labelsize': 'large'})
plt.rcParams.update({'legend.fontsize': 'medium'})

def mycolors():
    color_sequence = ['#1f77b4',  '#ff7f0e',  '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b',  '#e377c2',  '#7f7f7f',
                      '#bcbd22',  '#17becf']
    for color in color_sequence:
        yield color


def show():
    plt.show()


def save(picture_names):
    for index, figno in enumerate(plt.get_fignums()):
        try:
            plt.figure(figno)
            plt.savefig("%s.pdf" % picture_names[index], bbox_inches='tight')
        except ValueError:
            print("ValueError for %s", picture_names[index])

    plt.close('all')


def draw_frequency_distribution(data, dataset, type):
    print("Data:", data)
    fig = plt.figure(dpi=None, facecolor="white")
    fig.canvas.set_window_title("Frequency Distribution (%s)" % dataset)
    plt.clf()

    # Sort data by value then label
    for index, label in enumerate(sorted(data.keys())):
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


def new_figure(title):
    fig = plt.figure(dpi=None, facecolor="white")
    fig.canvas.set_window_title(title)
    plt.clf()
    plt.title(title)


def draw_record_time_curve(times, dataset, action):
    nplots = len(times)
    nrows = int((nplots - 1) / 2) + 1
    ncols = min(2, nplots)
    title = "%s time for a single record (%s)" % (action.capitalize(), dataset)
    fig = plt.figure(figsize=(8 * ncols, 5 * nrows), dpi=None, facecolor="white")
    fig.canvas.set_window_title(title)
    st = fig.suptitle("Dataset %s" % dataset)
    gs = gridspec.GridSpec(nrows, ncols)
    for index, run in enumerate(sorted(times.keys())):
        row = int(index - (index / 2))
        col = index % 2
        ax = fig.add_subplot(gs[row, col])
        draw_time_curve(times[run], action, run, ax)

    fig.tight_layout()
    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)


def draw_time_curve(times, action, run='', ax=plt):
    # Plot Precision-Recall curve
    # These are the colors that will be used in the plot
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    for index, indexer in enumerate(sorted(times.keys())):
        x_pos = np.arange(0, len(times[indexer]), 1)
        mean = np.mean(times[indexer])
        times_mean = [mean for i in x_pos]
        ax.plot(x_pos, times[indexer], lw=lw,
                color=color_sequence[2 * index + 1], label=indexer)
        # zorder assures that mean lines are always printed above
        ax.plot(x_pos, times_mean, color=color_sequence[2 * index],
                zorder=index + 100, linestyle='--')

    xlabel = "Record %s number" % action
    plt.xlabel(xlabel)
    plt.ylabel('Time (s)')
    plt.ylim([pow(10, -6), pow(10, 0)])
    plt.yscale('log')
    # plt.autoscale(enable=True, axis='y')
    plt.xlim([0, max(x_pos)])
    plt.legend(loc="best")
    plt.title("%s time for a single record (%s)" % (action.capitalize(), run))


def draw_precision_recall_curve(y_true, y_scores, dataset):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    draw_prc({'?': {'?': (precision, recall, thresholds)}}, dataset)


def draw_prc(prc_curves, dataset):
    nplots = len(prc_curves)
    nrows = int((nplots - 1) / 2) + 1
    ncols = min(2, nplots)
    fig = plt.figure(figsize=(8 * ncols, 5 * nrows), dpi=None, facecolor="white")
    fig.canvas.set_window_title("PRC (%s)" % dataset)
    st = fig.suptitle("Dataset %s" % dataset)
    color_sequence = ['#1f77b4',  '#ff7f0e',  '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b',  '#e377c2',  '#7f7f7f',
                      '#bcbd22',  '#17becf']
    # Plot Precision-Recall curve
    gs = gridspec.GridSpec(nrows, ncols)
    for index, run in enumerate(sorted(prc_curves.keys())):
        row = int(index - (index / 2))
        col = index % 2
        ax = fig.add_subplot(gs[row, col])
        for index, indexer in enumerate(sorted(prc_curves[run].keys())):
            precisions, recalls, thresholds = prc_curves[run][indexer]
            if (thresholds[0] == 0):
                recalls = recalls[1:]
                precisions = precisions[1:]
            ax.plot(recalls, precisions, 'yo-', lw=lw, label=indexer,
                    color=color_sequence[index], picker=True)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.02])
        plt.xlim([0.0, 1.02])
        plt.title("Precision-Recall Curve (%s)" % run)
        plt.legend(loc="lower left")

    fig.tight_layout()
    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)


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

    def autolabel(rects, ax):
        # Get y-axis height to calculate label position from.
        (y_bottom, y_top) = plt.ylim()
        y_height = y_top - y_bottom

        for rect in rects:
            height = rect.get_height()

            # Fraction of axis height taken up by this rectangle
            # p_height = (height / y_height)

            # If we can fit the label above the column, do that;
            # otherwise, put it inside the column.
            # if p_height > 0.95:  # arbitrary; 95% looked good to me.
            # @Problem: fitting label above the column confilct with 'best'
            # legend placement.
            label_position = height - (y_height * 0.05)
            # else:
            # label_position = height + (y_height * 0.01)

            ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                    str(round(height, 2)), ha='center', va='bottom')

    ax = plt.subplot(111)
    width = 0.28
    gap = 0.07
    offset = 0.
    color_series = mycolors()
    ticklabels = sorted(data.keys())
    ticks_pos = []
    for index, indexer in enumerate(sorted(data.keys())):
        start_offset = offset
        # Sort data by label
        for dataset, values in sorted(data[indexer].items(), key=lambda x: (x[0])):
            y_pos = width + offset
            offset += width
            color = next(color_series)
            bars = ax.bar(y_pos, values, width=width, align="center",
                          alpha=0.8, color=color, label=dataset)
            autolabel(bars, ax)

        ticks_pos.append(offset - ((len(data[indexer]) - 1) * (width  / 2)))
        offset += gap

    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(ticklabels)
    ax.legend(loc="bottom left")
