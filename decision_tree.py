import numpy as np
import pandas
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons


def plot(X, y, x_col, y_col):
    x_min = min(x_col) - 1
    x_max = max(x_col) + 1
    y_min = min(y_col) - 1
    y_max = max(y_col) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    print('xx: ', xx)
    print('yy: ', yy)

    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def make_tree(features, labels):
    print('features: ', features)
    print('labels: ', labels)
    global dt, tree_inside
    x_temp = []
    y_temp = []
    fitness = []

    # df = pandas.read_csv(address)

    # features = ['x', 'y', 'fitness']

    X = np.array(features)
    y = labels
    print('X: ', X)
    print('Y: ', y)
    # X = data[features].values
    # y = data['label'].values
    # X = data[:, :-1]
    # y = data[:, -1]

    # X, y = make_moons(noise=0.3, random_state=0)

    dt = DecisionTreeClassifier().fit(X, y)

    data = tree.export_graphviz(dt, out_file=None)
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')

    img = pltimg.imread('mydecisiontree.png')
    imgplot = plt.imshow(img)
    plt.show()

    tree_inside = dt.tree_
    print('lenX: ', len(X))
    for i in range(len(X)):
        x = np.array(X[i])
        print('x: ', x)
        res = f(0, x, dt.predict([x])[0])  # 0 is index of root node
        print('res: ', i, res)
        if len(res) > 0:
            ans = np.min([np.linalg.norm(x - n) for n in res])
            if ans <= 20:
                fitness.append(0)
            else:
                fitness.append(ans)

            print('ans: ', ans)
            # X[i][-1] = ans

        print('lenRes: ', len(res))

        # if res is not None and len(res) > 1 and y[i] != 0:
        #     X[i] = res[1]
        # elif len(res) == 1 and y[i] != 0:
        #     X[i] = res[0]

        # if res is not None and len(res) > 1:
        #     X[i] = res[1]
        # elif len(res) == 1:
        #     X[i] = res[0]
        # try:
        #     if y[i] != 0 and len(res) > 1:
        #         X[i] = res[1]
        # except:
        #     print('Something wrong!')

        x_temp.append(X[i][0])
        y_temp.append(X[i][1])
        # fit_temp.append(X[i][2])

    # new_val = []
    print('X', X)
    # for i in range(len(X)):
    #     new_val.append(X[i][:])
    #     new_val.append(y[i])
    plot(X, y, x_temp, y_temp)
    return list(zip(features, fitness))


def f(node, x, orig_label):
    if tree_inside.children_left[node] == tree_inside.children_right[node]:  # Meaning node is a leaf
        return [x] if dt.predict([x])[0] != orig_label else [None]

    if x[tree_inside.feature[node]] <= tree_inside.threshold[node]:
        orig = f(tree_inside.children_left[node], x, orig_label)
        xc = x.copy()
        xc[tree_inside.feature[node]] = tree_inside.threshold[node] + .01
        modif = f(tree_inside.children_right[node], xc, orig_label)
    else:
        orig = f(tree_inside.children_right[node], x, orig_label)
        xc = x.copy()
        xc[tree_inside.feature[node]] = tree_inside.threshold[node]
        modif = f(tree_inside.children_left[node], xc, orig_label)
    return [s for s in orig + modif if s is not None]

# make_tree("log/2021-02-18_08-03-17/test_data_sut_1.csv")
