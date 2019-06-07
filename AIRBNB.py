from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    data = pd.read_csv("train_user.csv", sep=";")

    #  On remplace tout les nan par d'autres valeurs
    data.age = data.age.fillna(value=int(data.age.median()), axis=0)  # pour l'age on remplace par la mediane
    data.first_affiliate_tracked = data.first_affiliate_tracked.fillna(
        value='unknown')  # pour les autres on remplace par unknown
    data.date_first_booking = data.date_first_booking.fillna(value='unknown')

    labEnc = []  # cette variable va stocker le label encoder de chaque colonne, cela permetra de décoder les données si besoin
    count = 0  # cette variable permet de se déplacer dans la variable label encoder

    for i in data:  # pour toutes les colonnes
        labEnc += [preprocessing.LabelEncoder()]  # on crée un nouveau label encoder
        labEnc[count].fit(data[i])  # on l'adapte à la colonne sur laquelle on travail
        data[i] = labEnc[count].transform(
            data[i])  # et on remplace les valeurs de la colonne de travail par les valeurs encodées
        count += 1

    # on sépare les données des cibles
    features = data.loc[:, data.columns[1:-1]]
    targets = data.loc[:, data.columns[-1]]

    # on normalise les données pour améliorer les résultats du model
    scaler = preprocessing.MinMaxScaler()
    normalizedFeatures = pd.DataFrame(scaler.fit_transform(features.values))

    # on sépare les données en un jeu d'entrainement et un jeu de test

    featuresTrain, featuresTest, targetsTrain, targetsTest = train_test_split(normalizedFeatures, targets,
                                                                              random_state=0, test_size=0.25)

    # on crée un modèle SVM et on l'entraine, on met peu de données car l'entrainement est long avec un grand nombre de données

    SVM = svm.SVC(gamma='scale')
    SVM.fit(featuresTrain[:20000], targetsTrain[:20000])

    #  on affiche le pourcentage de réussite

    print("Test set score: {:.2f}%".format(SVM.score(featuresTest[:20000], targetsTest[:20000]) * 100))

    # on fait des prédictions sur les 20 000 premieres données du pannel de test
    predictions = SVM.predict(featuresTest[:20000])

    # enfin on affiche la matrice de confusion
    print(confusion_matrix(targetsTest[:20000], predictions[:20000]))