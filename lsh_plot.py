import matplotlib.pyplot as plt
from simindex.helper import read_csv

dataset1 = read_csv("eval_lsh_restaurant.txt")
thresholds = sorted(list(set(x[0] for x in dataset1)))

fig = plt.figure(dpi=None, facecolor="white")
for threshold in thresholds:
    data = list(filter(lambda x: threshold == x[0], dataset1))
    num_perms = [x[1] for x in data]
    recall_score = [x[2] for x in data]
    label = "Threshold", threshold
    plt.plot(num_perms, recall_score, label=label)

plt.ylim(0, 1.05)
plt.legend(loc='best')
plt.show()
