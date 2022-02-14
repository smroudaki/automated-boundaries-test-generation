def fib(a):
    if a > 20:
        return "Perfect"
    if a > 10:
        return "Pass"
    else:
        return "Fail"


marks = {10, 15, 20, 5}
for item in marks:
    fib(item)
