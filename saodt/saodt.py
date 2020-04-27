import time
import numpy as np
import math
from numba import njit
from enum import Enum


class Movement(Enum):
    """ The Movement is a enum class represent a movement to generete a 
        neighbors in a solution of the metaheuristic
    """

    AUMENTAR = 1
    DIMINUIR = 2
    TROCAR = 3
    MINIMO = 4
    MAXIMO = 5
    ZERAR = 6
    INVERTE = 7


class DecisionNode:
    """ The Decision Node is a class represent a node in the decision tree
    """

    def __init__(
        self,
        column=-1,
        threshold=None,
        weights=np.array([]),
        is_leaf=None,
        results=np.array([]),
        children_left=None,
        children_right=None,
    ):
        """Initialize DecisionNode class

        Keyword Arguments:
            column {int} -- Column index where the threshold is located (default: {-1})
            threshold {float} -- The value of the threshold (default: {None})
            weights {np.ndarray} -- Array of the weights if used to oblique division (default: {np.array([])})
            is_leaf {bool} -- Indicate if node is leaf or not (default: {None})
            results {np.ndarray} -- If node is leaf represent the number of the each class (default: {np.array([])})
            children_left {DecisionNode} -- The left child of this node after the division (default: {None})
            children_right {DecisionNode} -- The right child of this node after the division (default: {None})
        """
        self.column = column
        self.threshold = threshold
        self.weights = weights
        self.is_leaf = is_leaf
        self.results = results
        self.children_left = children_left
        self.children_right = children_right


class SimulatedAnnealingObliqueDecisionTree:
    """The SimulatedAnnealingObliqueDecisionTree is a Decision Tree using a
        metaheuristic simulated annealing to oblique division of the nodes
    """

    def __init__(
        self,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        max_time=180,
        max_iterations=5000,
        initial_temperature=100,
        alpha=0.999,
    ):
        """Initialize a SimulatedAnnealingObliqueDecisionTreeClass

        Keyword Arguments:
            criterion {str} -- The criterion used for division nodes (default: {"entropy"})
            max_depth {int} -- The depth max of the decision tree (default: {None})
            min_samples_split {int} -- The samples min number in the node (default: {2})
            min_samples_leaf {int} -- The samples min number in leaf node (default: {1})
            min_impurity_decrease {float} -- The min number of the impurity for splitting node (default: {0.0})
            max_time {int} -- Max time in seconds for simulated annealing search for the solution (default: {60})
            max_iterations {int} -- Max iterations for simulated annealing search for the solution (default: {1000})
            initial_temperature {int} -- Initial Temperature of the simulated annealing algorithm (default: {100})
            alpha {float} -- Alpha of the simulated annealing algorithm (default: {0.999})
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_time = max_time
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.alpha = alpha

    @staticmethod
    @njit
    def entropy(classes):
        """calculates of the entropy (impurity criterion) in the node

        Arguments:
            classes {numpy.ndarray} -- number of occurrences of each class in the node

        Returns:
            float -- entropy calculated
        """
        total = np.sum(classes)
        entropy = 0.0

        for p in classes:
            if p == 0:
                continue
            else:
                p = p / total
                entropy -= p * np.log2(p)

        return entropy

    @staticmethod
    @njit
    def gini(classes):
        """calculates of the gini (impurity criterion) in the node

        Arguments:
            classes {numpy.ndarray} -- number of occurrences of each class in the node

        Returns:
            float -- gini calculated
        """
        total = np.sum(classes)
        gini = 0

        for p in classes:
            if p == 0:
                continue
            else:
                p = p / total
                gini += p * (1 - p)

        return gini

    @staticmethod
    def check_X_y(X, y):
        """checks if the data set is in the correct shape

        Arguments:
            X {numpy.ndarray} -- original dataset
            y {numpy.ndarray} -- original classes

        Raises:
            ValueError: If the dataset is empty
            ValueError: If the dataset is incorrect shape

        Returns:
            Tuple(numpy.ndarray, numpy.ndarray) -- return samples and classes if shape is correct
        """
        if len(X) <= 0 or len(y) <= 0:
            raise ValueError("Empty Database!")
        if len(np.shape(X)) != 2 or len(np.shape(y)) != 1:
            raise ValueError("Incorrect shape!")

        return X, y

    @staticmethod
    def build_weights(X):
        """build a random array of the weights

        Arguments:
            X {numpy.ndarray} -- Dataset whit calculates length of the weights

        Returns:
            numpy.ndarray -- Array of the weights
        """
        n_features = X.shape[1]
        rng = np.random.default_rng()

        return rng.random((n_features,))

    @staticmethod
    @njit
    def add_virtual_feature(X, weights):
        """Add a column containing a virtual feature calculated like a sum
        of the line with weights

        Arguments:
            X {numpy.ndarray} -- Dataset for add a column with virtual feature
            weights {numpy.ndarray} -- Array of the weights for calculation of the virtual feature

        Returns:
            numpy.ndarray -- Dataset containing a column with virtual feature
        """
        n_features = X.shape[1]
        X_aux = np.hstack((X, np.zeros((X.shape[0], 1))))
        cont = 0

        for sample in X:
            s = 0.0
            for i in range(len(sample)):
                s += sample[i] * weights[i]
            X_aux[cont, n_features] = s
            cont += 1

        return X_aux

    @staticmethod
    @njit
    def del_last_column(X):
        """delete a last column of the array

        Arguments:
            X {numpy.ndarray} -- Dataset for the remove last column

        Returns:
            numpy.ndarray -- Dataset without last colum
        """

        return X[:, :-1]

    @staticmethod
    def build_movement():
        """build a random movement for generate a solution

        Returns:
            Movement -- [description]
        """
        rng = np.random.default_rng()
        x = rng.integers(1030, size=1)[0]
        if x <= 349:
            return Movement.AUMENTAR
        elif x > 349 and x <= 699:
            return Movement.DIMINUIR
        elif x > 699 and x <= 989:
            return Movement.TROCAR
        elif x > 989 and x <= 904:
            return Movement.MINIMO
        elif x > 904 and x <= 999:
            return Movement.MAXIMO
        elif x > 999 and x <= 1004:
            return Movement.ZERAR
        elif x > 1004 and x <= 1029:
            return Movement.INVERTE

    @staticmethod
    def make_movement(weights, movement):
        """make a movement in weights, generating a neighbors weights

        Arguments:
            weights {numpy.ndarray} -- array of the weights
            movement {Movement} -- The movement to be applied in the weights array

        Returns:
            numpy.ndarray -- Weights with movement applied
        """
        weights = np.copy(weights)
        rng = np.random.default_rng()
        col_modified = rng.integers(weights.shape[0], size=1)[0]

        if movement == Movement.AUMENTAR:
            if weights[col_modified] == 0:
                weights[col_modified] += rng.random()
            else:
                weights[col_modified] += 0.1 * weights[col_modified]

        elif movement == Movement.DIMINUIR:
            if weights[col_modified] == 0:
                weights[col_modified] -= rng.random()
            else:
                weights[col_modified] -= 0.1 * weights[col_modified]

        elif movement == Movement.TROCAR:
            col_swap = rng.integers(weights.shape[0], size=1)[0]

            while col_swap == col_modified:
                col_swap = rng.integers(weights.shape[0], size=1)[0]

            weights[col_modified], weights[col_swap] = (
                np.copy(weights[col_swap]),
                np.copy(weights[col_modified]),
            )

        elif movement == Movement.MINIMO:
            weights[col_modified] = -1

        elif movement == Movement.MAXIMO:
            weights[col_modified] = 1

        elif movement == Movement.ZERAR:
            weights[col_modified] = 0

        elif movement == Movement.INVERTE:
            weights[col_modified] = -1 * weights[col_modified]

        return weights

    def get_params(self, deep=True):
        """Return a dictionary with params from class

        Returns:
            Dict -- Dictionary containing the params of the class
        """
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_time": self.max_time,
            "max_iterations": self.max_iterations,
            "initial_temperature": self.initial_temperature,
            "alpha": self.alpha,
        }

    def set_params(self, **parameters):
        """modified of params in the class

        Returns:
            SimulatedAnnealingObliqueDecisionTree -- return the class
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """build a decision tree classifier from the training set

        Arguments:
            X {numpy.ndarray} -- Original dataset for fit
            y {numpy.ndarray} -- Original classes for fit

        Returns:
            SimulatedAnnealingObliqueDecisionTree -- return the class
        """
        X, y = self.check_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_ = X.shape[1]
        self.tree_ = self.make_tree(X, y)

        return self

    def best_in_column(self, X, y, index):
        """Return the best impurity in column selected

        Arguments:
            X {numpy.ndarray} -- dataset for to evaluate the best impurity
            y {numpy.ndarray} -- classes for to evaluate the best impurity
            index {int} -- Index of the column to be analyzed

        Returns:
            Tuple -- A tuple containing of the threshold value and the best impurity value
        """
        best_gain = np.iinfo(np.int).min
        best_gini = np.iinfo(np.int).max
        best_threshold = None
        classes, count_classes = np.unique(y, return_counts=True)

        if self.criterion == "entropy":
            total_impurity = self.entropy(count_classes)
        else:
            total_impurity = self.gini(count_classes)

        count_right = np.zeros(self.n_classes_)
        count_right[classes] = count_classes
        count_left = np.zeros(self.n_classes_)
        sum_right = np.sum(count_right)
        sum_left = 0

        column = np.array([[i, v] for i, v in enumerate(list(X[:, index]))])
        column = column[np.argsort(column[:, 1])]
        n_samples = X.shape[0]
        i = 0
        while i < n_samples:
            count_right[y[int(column[i, 0])]] -= 1
            count_left[y[int(column[i, 0])]] += 1
            count = 0
            j = i + 1
            if j < n_samples:
                while column[i, 1] == column[j, 1]:
                    count_right[y[int(column[j, 0])]] -= 1
                    count_left[y[int(column[j, 0])]] += 1
                    count += 1
                    j += 1
                    if j >= n_samples:
                        break

            sum_right -= count + 1
            sum_left += count + 1
            p1 = sum_left / float(n_samples)
            p2 = sum_right / float(n_samples)

            if self.criterion == "entropy":
                info_gain = (
                    total_impurity
                    - (p1 * self.entropy(count_left))
                    - (p2 * self.entropy(count_right))
                )

                if info_gain > best_gain and sum_left >= 0 and sum_right >= 0:
                    best_gain = info_gain
                    best_threshold = column[i, 1]
            else:
                index_gini = self.gini(count_left) ** 2 + self.gini(count_right) ** 2
                if index_gini < best_gini and sum_left >= 0 and sum_right >= 0:
                    best_gini = index_gini
                    best_threshold = column[i, 1]

            i += count + 1

        if self.criterion == "entropy":
            return best_threshold, best_gain
        else:
            return best_threshold, best_gini

    def simulated_annealing(self, X, y):
        """simulated annealing metaheuristic for the weights solution

        Arguments:
            X {numpy.ndarray} -- Dataset
            y {numpy.ndarray} -- Classes

        Returns:
            Tuple -- A tuple containing final X with virtual feature, the column of the best threshold,  the value and the best threshold and the best solution weights
        """
        best_gain = np.iinfo(np.int).min
        best_gini = np.iinfo(np.int).max
        best_threshold = None
        n_features = X.shape[1]
        t = self.initial_temperature
        weights = self.build_weights(X)
        modified = False
        impurity_current = 0
        X_with_virtual_feature = None
        rng = np.random.default_rng()
        iteration = 0
        start_time = time.time()

        while (
            time.time() - start_time < self.max_time and iteration < self.max_iterations
        ):
            if iteration == 0 or modified:
                modified = False
                X_with_virtual_feature = self.add_virtual_feature(X, weights)
                _, impurity_current = self.best_in_column(
                    X_with_virtual_feature, y, n_features
                )

            weights_neighbors = self.make_movement(weights, self.build_movement())
            X_with_virtual_feature = self.add_virtual_feature(X, weights_neighbors)
            _, impurity_test = self.best_in_column(
                X_with_virtual_feature, y, n_features
            )

            if self.criterion == "entropy":
                if impurity_test < impurity_current:
                    delta = impurity_current - impurity_test
                    x = rng.random()
                    if x <= math.e ** ((-delta) / t):
                        weights = np.copy(weights_neighbors)
                        modified = True
                else:
                    weights = np.copy(weights_neighbors)
                    modified = True
            else:
                if impurity_test > impurity_current:
                    delta = impurity_test - impurity_current
                    x = rng.random()
                    if x <= math.e ** ((-delta) / t):
                        weights = np.copy(weights_neighbors)
                        modified = True
                else:
                    weights = np.copy(weights_neighbors)
                    modified = True

            t *= self.alpha
            iteration += 1

        X_final = self.add_virtual_feature(X, weights)
        index = -1

        for i in range(X_final.shape[1]):
            aux_threshold, aux_gain = self.best_in_column(X_final, y, i)
            if self.criterion == "entropy":
                if aux_gain > best_gain:
                    best_gain = aux_gain
                    best_threshold = aux_threshold
                    index = i
            else:
                if aux_gain < best_gini:
                    best_gini = aux_gain
                    best_threshold = aux_threshold
                    index = i

        return X_final, index, best_threshold, weights

    def stopping_criterion(self, n_samples, n_classes, depth):
        """tests whether to stop building the tree at that node

        Arguments:
            n_samples {int} -- number of the samples in the node
            n_classes {int} -- number of the classes int the node
            depth {int} -- level the node in the tree

        Returns:
            bool -- return true if the any criterion satisfied and false if the not satisfied
        """
        return (
            self.max_depth == depth
            or n_classes == 1
            or n_samples <= self.min_samples_split
            or n_samples / 2 == self.min_samples_leaf
        )

    def make_tree(self, X, y, depth=1):
        """help function for the construct tree

        Arguments:
            X {numpy.ndarray} -- Dataset
            y {numpy.ndarray} -- Classes

        Keyword Arguments:
            depth {int} -- Level of the node (default: {1})

        Returns:
            DecisionNode -- return a node
        """
        if len(X) == 0 or len(y) == 0:
            return DecisionNode()
        classes, count_classes = np.unique(y, return_counts=True)
        n_samples = X.shape[0]
        n_classes = classes.shape[0]

        if not self.stopping_criterion(n_samples, n_classes, depth):
            X_aux, index, threshold, weights = self.simulated_annealing(X, y)
            div = X_aux[:, index] <= threshold
            X_left = self.del_last_column(X_aux[div])
            X_right = self.del_last_column(X_aux[~div])
            c_left = self.make_tree(X_left, y[div], depth + 1)
            c_right = self.make_tree(X_right, y[~div], depth + 1)

            return DecisionNode(
                column=index,
                threshold=threshold,
                weights=weights,
                children_left=c_left,
                children_right=c_right,
            )

        values = np.zeros(self.n_classes_)
        values[classes] = count_classes
        return DecisionNode(is_leaf=True, results=values)

    def predict(self, X):
        """predict a classes of the array for characteristics

        Arguments:
            X {numpy.ndarray} -- array of the characteristics to be predict

        Returns:
            classify -- help function for build decision tree
        """
        return self.classify(self.tree_, X)

    def classify(self, node, X):
        """hel funtion for build decision tree

        Arguments:
            node {DecisionNode} -- node to be observed
            X {numpy.ndarray} -- array of the characteristics to be predict

        Returns:
            numpy.ndarray -- a array containing prediction class
        """

        if node.is_leaf:
            return np.zeros(X.shape[0]) + np.argmax(node.results)

        X = self.add_virtual_feature(X, node.weights)
        div = X[:, node.column] <= node.threshold
        y_pred = np.zeros(X.shape[0])

        X_left = self.del_last_column(X[div])
        X_right = self.del_last_column(X[~div])

        if div.sum() > 0:
            y_pred[div] = self.classify(node.children_left, X_left)

        if (~div).sum() > 0:
            y_pred[~div] = self.classify(node.children_right, X_right)

        return y_pred


from reader_csv import read_csv


X, y = read_csv("iris.csv", "class")
clf = SimulatedAnnealingObliqueDecisionTree()
clf = clf.fit(X, y)
result = clf.predict(X)
print(result == y)
