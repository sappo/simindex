import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import precision_recall_curve, recall_score, precision_score

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

y_true = np.array([0, 0, 1, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, -1.0])


def draw_precision_recall_curve(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if (thresholds[0] == 0):
        recall = recall[1:]
        precision = precision[1:]
    print(thresholds)
    print(recall)
    print(precision)

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, lw=lw, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example')
    plt.legend(loc="lower left")
    plt.show()

# draw_precision_recall_curve(y_true, y_scores)

# y_true   = np.array([1  , 1,   1,   0,   0  , 1  , 1  ])
# y_scores = np.array([0.8, 0.7, 0.5, 0.3, 0.6, 0.0, 0.0])
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# print("T1:",thresholds)
# print("R1:", recall)
# print("P1:", precision)

# y_true   = np.array([1  , 1,   1,   0,   0  , 1  , 1  , 0  , 0  , 0  ])
# y_scores = np.array([0.8, 0.7, 0.5, 0.3, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
# print("T2:", thresholds)
# print("R2:", recall)
# print("P2:", precision)

# # TP = 5, FP = 2, FN = 3, TN = 2
# y_true_wo = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
# y_pred_wo = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

# y_true_wi = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
# y_pred_wi = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# print(precision_score(y_true_wo, y_pred_wo))
# print(precision_score(y_true_wo, y_pred_wo))
# print(precision_score(y_true_wi, y_pred_wi))
# print(precision_score(y_true_wi, y_pred_wi))

# print(recall_score(y_true_wo, y_pred_wo))
# print(recall_score(y_true_wo, y_pred_wo))
# print(recall_score(y_true_wi, y_pred_wi))
# print(recall_score(y_true_wi, y_pred_wi))
