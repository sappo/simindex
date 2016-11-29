from simindex import DySimII
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
from plot import draw_precision_recall_curve

def bad_encode(a):
    return a[:1]

s = DySimII(2, encode_fn=bad_encode, threshold=0.,
            gold_standard="restaurant_gold.csv",
            gold_attributes=['id_1', 'id_2'])

s.insert_from_csv("restaurant.csv", ["id", "name", "addr"])
recall = s.recall()

result, y_true1, y_scores, y_true2, y_pred = s.query_from_csv('restaurant.csv', ["id", "name", "addr"])
print("Query records:", len(result))
print("Recall blocking:", recall)
print("P1:", precision_score(y_true2, y_pred))
print("R1:", recall_score(y_true2, y_pred))

# print(y_scores)
draw_precision_recall_curve(y_true1, y_scores)
