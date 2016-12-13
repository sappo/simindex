import matplotlib.pyplot as plt
from simindex.helper import read_csv

dataset1 = read_csv("evaluation/plot-len550-min2-max385-step1.txt")
ds1_n_ranges = [x[0] for x in dataset1]
ds1_kmeans_time = [x[1] for x in dataset1]
ds1_sil_sc = [x[2] for x in dataset1]
ds1_sil_time = [x[3] for x in dataset1]
ds1_recall_sc = [x[4] for x in dataset1]
ds1_recall_time = [x[5] for x in dataset1]

restaurant = read_csv("evaluation/plot-len864-min2-max604-step1.txt")
res_n_ranges = [x[0] for x in restaurant]
res_kmeans_time = [x[1] for x in restaurant]
res_sil_sc = [x[2] for x in restaurant]
res_sil_time = [x[3] for x in restaurant]
res_recall_sc = [x[4] for x in restaurant]
res_recall_time = [x[5] for x in restaurant]

febrl_4k = read_csv("evaluation/plot-len5001-min2-max3500-step8.txt")
f4k_n_ranges = [x[0] for x in febrl_4k]
f4k_kmeans_time = [x[1] for x in febrl_4k]
f4k_sil_sc = [x[2] for x in febrl_4k]
f4k_sil_time = [x[3] for x in febrl_4k]
f4k_recall_sc = [x[4] for x in febrl_4k]
f4k_recall_time = [x[5] for x in febrl_4k]

fig = plt.figure(dpi=None, facecolor="white")

plt.subplot(2, 1, 1)  # (rows, columns, panel number)
plt.title("Silhouette-Recall Curve")
plt.ylabel("Score [0,1]")
plt.xlabel("K-Means k value")
plt.plot(ds1_n_ranges, ds1_sil_sc, label="DS1 silhouette score")
plt.plot(ds1_n_ranges, ds1_recall_sc, label="DS1 recall score")
plt.plot(res_n_ranges, res_sil_sc, label="RES silhouette score")
plt.plot(res_n_ranges, res_recall_sc, label="RES recall score")
plt.plot(f4k_n_ranges, f4k_sil_sc, label="F4k silhouette score")
plt.plot(f4k_n_ranges, f4k_recall_sc, label="F4k recall score")
plt.legend(loc='best')

# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.title("Silhouette-Recall Times")
plt.ylabel("Seconds (s)")
plt.xlabel("K-Means k value")
plt.plot(ds1_n_ranges, ds1_kmeans_time, label="DS1 kmeans time")
plt.plot(ds1_n_ranges, ds1_sil_time, label="DS1 silhouette time")
plt.plot(res_n_ranges, res_kmeans_time, label="RES kmeans time")
plt.plot(res_n_ranges, res_sil_time, label="RES silhouette time")
plt.plot(f4k_n_ranges, f4k_kmeans_time, label="F4k kmeans time")
plt.plot(f4k_n_ranges, f4k_sil_time, label="F4k silhouette time")
plt.legend(loc='auto')

plt.show()
