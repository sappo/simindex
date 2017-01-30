import pandas as pd

restaurant_records = [
    # P (0, 1), (2, 3), (4, 5), (6, 7)
    ["0", "mario's pizza", "italian"],
    ["1", "marios pizza", "italian"],
    ["2", "fringal", "french bistro"],
    ["3", "fringale", "french bistro"],
    ["4", "yujean kang's gourmet cuisine", "asian"],
    ["5", "yujean kang's best cuisine", "asian"],
    ["6", "big belly burger", "american"],
    ["7", "big belly burger", "german"],
    # n (8, 9), (10, 11), (12, 13), (14, 15)
    ["8", "sally's cafè and internet", "tex-mex cafe"],
    ["9", "cafè sunflower and more", "health food"],
    ["10", "roosevelt tamale parlor", "mexican"],
    ["11", "wa-ha-ka oaxaca moaxo", "mexican"],
    ["12", "thailand restaurant", "thai"],
    ["13", "andre's petit restaurant", "spanish"],
    ["14", "zubu", "japanese"],
    ["15", "nobu", "japanese"],
]

restaurant_gold_pairs = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7)
]

restaurant_dataset = {x[0]: x[1:] for x in restaurant_records}

restaurant_df = pd.DataFrame([[r[1], r[2]] for r in restaurant_records],
                             index=[r[0] for r in restaurant_records],
                             columns=["name", "kind"])

restaurant_store = pd.HDFStore("restaurant.h5",
                               driver="H5FD_CORE",
                               driver_core_backing_store=0)
restaurant_store.append("restaurant", restaurant_df)
