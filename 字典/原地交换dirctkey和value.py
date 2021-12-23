def func(dirct):
    dirct = {y : x for x, y in dirct.items()}
    print(dirct)

func({1: 'A', 2: 'B', 3: 'C', 4: 'D'})