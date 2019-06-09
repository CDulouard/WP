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

    age_distribution_lesseq_50k = lesseq_50k["age"].value_counts()
    age_distribution_greater_50k = greater_50k["age"].value_counts()
    age_distribution = data["age"].value_counts()

    color_coefs = [(int(
        age_distribution_greater_50k[i]) * 100 / int(
        age_distribution[i]) / 100) if i in age_distribution_greater_50k.keys() else 0
                   for i in age_distribution.keys()]
    print(color_coefs)

    colors = [(i * 2, 0, 1 / (3 + i)) for i in color_coefs]

    plt.scatter(age_distribution.keys(), age_distribution, c=colors, alpha=1, label="Total")

    plt.scatter(age_distribution_greater_50k.keys(), age_distribution_greater_50k, color="red", alpha=0.7, marker="x",
                label=">50K")

    plt.scatter(age_distribution_lesseq_50k.keys(), age_distribution_lesseq_50k, color="blue", alpha=0.7, marker="x",
                label="<=50K")

    plt.title('Age distribution per incomes ')
    plt.xlabel('age')
    plt.ylabel('Number of entries')
    plt.legend(numpoints=1)
    plt.show()

    """
    Number of entries per incomes depending on the education
    """

    level_educ_distrib = data["education"].value_counts()

    print(level_educ_distrib)

    bins = [x + 0.5 for x in range(-1, len(level_educ_distrib))]
    plt.hist([lesseq_50k["education"], greater_50k["education"]], bins=bins, color=['blue', 'red'],
             label=["Less than 50K", "Greater than 50K"],
             histtype='barstacked', orientation='vertical', rwidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel('Number of entries')
    plt.xlabel('Education')
    plt.title('Education ditribution per income')
    plt.legend()
    plt.show()

    """
    Number of entries per incomes depending on the marital status
    """

    marital_status_distrib = data["marital-status"].value_counts()

    print(marital_status_distrib)

    bins = [x + 0.5 for x in range(-1, len(marital_status_distrib))]
    plt.hist([lesseq_50k["marital-status"], greater_50k["marital-status"]], bins=bins, color=['blue', 'red'],
             label=["Less than 50K", "Greater than 50K"],
             histtype='barstacked', orientation='vertical', rwidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel('Number of entries')
    plt.xlabel('Marital status')
    plt.title('Marital status distribution per income')
    plt.legend()
    plt.show()

    """
    Number of entries per incomes depending on the occupation
    """

    occupation_distrib = data["occupation"].value_counts()

    print(occupation_distrib)

    bins = [x + 0.5 for x in range(-1, len(occupation_distrib))]
    plt.hist([lesseq_50k["occupation"], greater_50k["occupation"]], bins=bins, color=['blue', 'red'],
             label=["Less than 50K", "Greater than 50K"],
             histtype='barstacked', orientation='vertical', rwidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel('Number of entries')
    plt.xlabel('Occupation')
    plt.title('Occupation distribution per income')
    plt.legend()
    plt.show()

    """
    Number of entries per incomes depending on the relationship
    """

    relationship_distrib = data["relationship"].value_counts()

    print(relationship_distrib)

    bins = [x + 0.5 for x in range(-1, len(relationship_distrib))]
    plt.hist([lesseq_50k["relationship"], greater_50k["relationship"]], bins=bins, color=['blue', 'red'],
             label=["Less than 50K", "Greater than 50K"],
             histtype='barstacked', orientation='vertical', rwidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel('Number of entries')
    plt.xlabel('Relationship')
    plt.title('Relationship distribution per income')
    plt.legend()
    plt.show()

    """
    Number of entries per incomes depending on the race
    """

    race_distrib = data["race"].value_counts()

    print(race_distrib)

    bins = [x + 0.5 for x in range(-1, len(race_distrib))]
    plt.hist([lesseq_50k["race"], greater_50k["race"]], bins=bins, color=['blue', 'red'],
             label=["Less than 50K", "Greater than 50K"],
             histtype='barstacked', orientation='vertical', rwidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel('Number of entries')
    plt.xlabel('Race')
    plt.title('Race distribution per income')
    plt.legend()
    plt.show()

    """
    Number of entries per incomes depending on the sex
    """

    sex_distrib = data["sex"].value_counts()

    print(sex_distrib)

    bins = [x + 0.5 for x in range(-1, len(sex_distrib))]
    plt.hist([lesseq_50k["sex"], greater_50k["sex"]], bins=bins, color=['blue', 'red'],
             label=["Less than 50K", "Greater than 50K"],
             histtype='barstacked', orientation='vertical', rwidth=0.8)
    plt.xticks(rotation=90)
    plt.ylabel('Number of entries')
    plt.xlabel('Sex')
    plt.title('Sex distribution per income')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
