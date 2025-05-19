import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ici on simule un dataset simplifié avec des variables explicatives
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),  # revenu annuel
    'balance': np.random.normal(15000, 5000, n_samples),  # solde moyen du compte
    'num_products': np.random.randint(1, 5, n_samples),   # nombre de produits bancaires
    'credit_score': np.random.randint(300, 850, n_samples), # score de crédit
})

#affichage
print("Aperçu des données :")
print(data.head())

# Variable cible : montant déposé (dépend fortement de revenu et solde)
data['deposit_amount'] = (
    data['income'] * 0.05 +
    data['balance'] * 0.1 +
    data['credit_score'] * 10 +
    np.random.normal(0, 1000, n_samples)  # bruit aléatoire
)
print(data.head())


# 4. Séparation des données en train/test
X = data.drop('deposit_amount', axis=1)
y = data['deposit_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)
# 6. Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# 7. Évaluation des performances
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE (Erreur absolue moyenne) : {mae:.2f}")
print(f"RMSE (Racine de l'erreur quadratique moyenne) : {rmse:.2f}")
print(f"R² (Coefficient de détermination) : {r2:.2f}")


# 8. Visualisation des prédictions vs valeurs réelles
import matplotlib
matplotlib.use('Agg')  # Backend non interactif, pas d'affichage de fenêtre

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Régression linéaire : Prédictions vs Valeurs réelles")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ligne y=x
plt.savefig("predictions_vs_reelles.png")  # Sauvegarde la figure 
