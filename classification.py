from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
iris = load_iris(as_frame=True)
X = iris.data  # Données d'entrée : mesures de fleurs
y = iris.target  # Étiquettes : classes (0, 1, 2)

# Aperçu
iris_df = iris.frame
iris_df['target'] = iris.target_names[y]
# Affichage
print("Aperçu des données :")
print(iris_df.head())

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# AALGO de classification Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
clf.fit(X_train,y_train)

# Prédictions sur l'ensemble de test
y_pred = clf.predict(X_test)

# Évaluation du modèle
from sklearn.metrics import classification_report, confusion_matrix
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print("Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
