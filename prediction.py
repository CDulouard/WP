from typing import List
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree

WORK_FILE: str = "datasets/adult.data"
COLUMNS: List[str] = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                      "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                      "hours-per-week", "native-country", "income"]


def main():
    data: pd.DataFrame = pd.read_csv(WORK_FILE, sep=',', names=COLUMNS)  # importing data
    print(str(data.keys()) + "\ndata shape : " + str(data.shape))  # print infos of the DataFrame

    """
    Encode each columns of our datas and store the label encoders in a list to use it
    later if we need to decode the columns
    """

    encoded_data: pd.DataFrame = data.copy()
    lab_enc: List[preprocessing.LabelEncoder] = []
    for i, e in enumerate(data):
        lab_enc += [preprocessing.LabelEncoder()]
        lab_enc[i].fit(data[e])
        encoded_data[e] = lab_enc[i].transform(data[e])
    print("\nencoded_data shape : " + str(encoded_data.shape))  # check the shape of the new DataFrame

    features: pd.DataFrame = encoded_data.loc[:, encoded_data.columns[0:-1]]
    targets: pd.DataFrame = encoded_data.loc[:, encoded_data.columns[-1]]

    print("\n\n" + str(features.keys()) + "\nfeatures shape : " + str(features.shape))

    """
    Normalization of the features to try to improve the score of the prediction
    """

    scaler: preprocessing.MinMaxScaler = preprocessing.MinMaxScaler()
    normalized_features: pd.DataFrame = pd.DataFrame(scaler.fit_transform(features.values))

    """
    Split features and targets into a train and a test data set
    """
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets,
                                                                                  random_state=0, test_size=0.25)

    normalized_features_train, normalized_features_test, normalized_targets_train, normalized_targets_test = \
        train_test_split(
            normalized_features, targets,
            random_state=0, test_size=0.25)

    """
    Train and test a SVM to predict the targets
    Works better with normalized values
    """
    svm_model: svm.SVC = svm.SVC(gamma='scale')
    svm_model.fit(normalized_features_train[:20000], normalized_targets_train[:20000])

    print("\n\nSVM :\nTest set score: {:.2f}%".format(
        svm_model.score(normalized_features_test[:20000], normalized_targets_test[:20000]) * 100))

    """
    Train and test a Stochastic Gradient Descent model to predict the targets
    Works better with normalized values
    """

    sdg_model: SGDClassifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000000)
    sdg_model.fit(normalized_features_train, normalized_targets_train)
    print("\n\nSDG :\nTest set score: {:.2f}%".format(
        sdg_model.score(normalized_features_test, normalized_targets_test) * 100))

    """
    Train and test a Tree model to predict the targets
    """

    tree_model: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier()
    tree_model.fit(features_train, targets_train)
    print("\n\nTree :\nTest set score: {:.2f}%".format(
        tree_model.score(features_test, targets_test) * 100))


if __name__ == "__main__":
    main()
