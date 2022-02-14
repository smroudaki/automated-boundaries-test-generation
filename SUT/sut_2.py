import math


def testfunc_2(x, y):
    if -10 * math.exp((-math.fabs(x) - math.fabs(y))) - 7 * math.exp(
            (-math.fabs(x - 4) - math.fabs(y))) - 19 * math.exp(-math.fabs(x + 10) - math.fabs(y - 5)) + math.sin(
            2 * x * y) + 5 <= 0:
        print("ok")
        return "answer"


def main(x, y):
    return testfunc_2(x, y)
