from functools import wraps

def dec(func):
    @wraps(func)
    def decoratetionfun(*args, **kwargs):
        print("123")
        func()
        print("apple")

    return decoratetionfun

@dec
def a():
    print("una")

a()
# a()

