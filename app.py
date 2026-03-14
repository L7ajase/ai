import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# 1. Chargement des données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
cols = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment', 'inst_rate', 'sex', 'debtors', 'residence', 'property', 'age', 'plans', 'housing', 'credits', 'job', 'liable', 'tel', 'foreign', 'target']
df = pd.read_csv(url, sep=' ', names=cols)
df['target'] = df['target'].replace({1: 1, 2: 0})

# 2. Pipeline de préparation
X = pd.get_dummies(df.drop('target', axis=1), drop_first=True)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Entraînement
model = LogisticRegression()
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

# 4. Interface Streamlit
st.title("Simulateur de Scoring Bancaire")
st.write("Cette interface permet de visualiser l'impact du seuil de décision sur la performance du modèle.")

# Affichage du DataFrame
if st.checkbox("Afficher le jeu de données"):
    st.dataframe(df.head(10))

# Curseur pour le seuil
seuil = st.sidebar.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.05)

# Calcul et Affichage Matrice de confusion
y_pred = (probs > seuil).astype(int)
cm = confusion_matrix(y_test, y_pred)

st.write(f"### Matrice de Confusion (Seuil actuel: {seuil})")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Prédictions')
ax.set_ylabel('Valeurs réelles')
st.pyplot(fig)