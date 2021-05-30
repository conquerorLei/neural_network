from unit_7_practice import total as t

if __name__ == "__main__":
    index_property = [0, 1, 2]
    iris_train, iris_test = t.getDataSet()
    x_train, y_train, x_test, y_test = t.getData(iris_train, iris_test, index_property=index_property, not_index_species=1)
    mx_train, my_train, mx_test, my_test = t.normalize(x_train, y_train, x_test, y_test)
    ce_train, ce_test, acc_train, acc_test, w = t.trainModel(mx_train, my_train, mx_test, my_test, learn_rate=0.015, rand=4, my_iter=120)
    t.show(ce_train, ce_test, acc_train, acc_test)
