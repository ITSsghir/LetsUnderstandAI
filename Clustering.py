# 📦 Import des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 🎲 Génération de données simulées
np.random.seed(42)
n_clients = 200

# Colonnes simulées
ages = np.random.randint(18, 70, size=n_clients)
revenus = np.random.normal(loc=40000, scale=15000, size=n_clients).astype(int)
soldes = np.random.normal(loc=15000, scale=10000, size=n_clients).astype(int)
scores_credit = np.random.randint(300, 850, size=n_clients)

# Création du DataFrame
df = pd.DataFrame({
    "Âge": ages,
    "Revenu_annuel": revenus,
    "Solde_moyen": soldes,
    "Score_credit": scores_credit
})

# 2. ⚙️ Standardisation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. 🤖 Clustering avec K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# Ajout des clusters dans le DataFrame
df['Cluster'] = clusters

# 4. 📊 Visualisation
import matplotlib
matplotlib.use('Agg')  # Backend non interactif, pas d'affichage de fenêtre
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Revenu_annuel", y="Solde_moyen", hue="Cluster", palette="Set2")
plt.title("Segmentation des clients bancaires (Clustering K-Means)")
plt.xlabel("Revenu annuel (€)")
plt.ylabel("Solde moyen (€)")
plt.grid(True)
plt.tight_layout()
plt.savefig("clustering.png")  # Sauvegarde la figure 

# 5. 👀 Affichage d’un aperçu des données
print(df.head())

# 6. 🔍 Analyse des clusters
print("\nRésumé par cluster :")
print(df.groupby('Cluster').mean(numeric_only=True))
