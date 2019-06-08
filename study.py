from typing import List
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

WORK_FILE: str = "datasets/adult.data"
COLUMNS: List[str] = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                      "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                      "hours-per-week", "native-country", "income"]


def main():
    data: pd.DataFrame = pd.read_csv(WORK_FILE, sep=',', names=COLUMNS)  # importing data
    print(str(data.keys()) + "\ndata shape : " + str(data.shape))  # print infos of the DataFrame

    """
    Split the DataFrame into two DF, one for >50k and an other one for <=50K
    """
    greater_50k: pd.DataFrame = data.query('income == " >50K" ').loc[:, data.columns[0:-1]]
    lesseq_50k: pd.DataFrame = data.query('income == " <=50K"').loc[:, data.columns[0:-1]]

    """
    Age distribution calculation and plot
    """

    # age_distribution_lesseq_50k = lesseq_50k["age"].value_counts()
    # age_distribution_greater_50k = greater_50k["age"].value_counts()
    # age_distribution = data["age"].value_counts()
    #
    # color_coefs = [(int(
    #     age_distribution_greater_50k[i]) * 100 / int(
    #     age_distribution[i]) / 100) if i in age_distribution_greater_50k.keys() else 0
    #                for i in age_distribution.keys()]
    # print(color_coefs)
    #
    # colors = [(i * 2, 0, 1 / (3 + i)) for i in color_coefs]
    #
    # plt.scatter(age_distribution.keys(), age_distribution, c=colors, alpha=1, label="Total")
    #
    # plt.scatter(age_distribution_greater_50k.keys(), age_distribution_greater_50k, color="red", alpha=0.7, marker="x",
    #             label=">50K")
    #
    # plt.scatter(age_distribution_lesseq_50k.keys(), age_distribution_lesseq_50k, color="blue", alpha=0.7, marker="x",
    #             label="<=50K")
    #
    # plt.title('Age distribution')
    # plt.xlabel('age')
    # plt.ylabel('count')
    # plt.legend(numpoints=1)
    # plt.show()


        

if __name__ == "__main__":
    main()
