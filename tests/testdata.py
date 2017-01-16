restaurant_records = [
    # P (0, 1), (2, 3), (4, 5), (6, 7)
    ["0", "Mario's Pizza", "Italian"],
    ["1", "Marios Pizza", "Italian"],
    ["2", "Fringal", "French Bistro"],
    ["3", "Fringale", "French Bistro"],
    ["4", "Yujean Kang's Gourmet Cuisine", "Asian"],
    ["5", "Yujean Kang's Best Cuisine", "Asian"],
    ["6", "Big Belly Burger", "American"],
    ["7", "Big Belly Burger", "German"],
    # N (8, 9), (10, 11), (12, 13), (14, 15)
    ["8", "Sally's Cafè and Internet", "Tex-Mex Cafe"],
    ["9", "Cafè Sunflower and More", "Health Food"],
    ["10", "Roosevelt Tamale Parlor", "Mexican"],
    ["11", "Wa-Ha-Ka Oaxaca Moaxo", "Mexican"],
    ["12", "Thailand Restaurant", "Thai"],
    ["13", "Andre's Petit Restaurant", "Spanish"],
    ["14", "Zubu", "Japanese"],
    ["15", "Nobu", "Japanese"],
]

restaurant_dataset = {x[0]: x[1:] for x in restaurant_records}
