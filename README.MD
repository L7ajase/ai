# Simulateur de Scoring Bancaire

Un système de scoring de crédit basé sur la **régression logistique**, avec une interface interactive développée avec **Streamlit**, permettant de simuler et d'analyser l'impact des seuils de décision sur la classification des risques bancaires.

---

##  Description du Projet

Ce projet vise à évaluer la **solvabilité des clients d'une banque** à partir du jeu de données [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) (UCI Machine Learning Repository). Il repose sur un pipeline complet allant du prétraitement des données jusqu'à la visualisation interactive des résultats du modèle.

L'objectif principal est de fournir un outil simple et intuitif permettant à un utilisateur métier ou technique de :
- Comprendre le comportement d'un modèle de scoring,
- Simuler l'effet du seuil de décision sur les prédictions,
- Visualiser en temps réel la matrice de confusion.

---

##  Fonctionnalités

- ✅ **Chargement automatique des données** depuis le dépôt UCI
- ✅ **Prétraitement complet** : encodage des variables catégorielles (One-Hot Encoding) et normalisation (StandardScaler)
- ✅ **Entraînement d'un modèle de régression logistique**
- ✅ **Interface interactive Streamlit** avec :
  - Affichage optionnel du jeu de données (10 premières lignes)
  - Curseur (slider) pour ajuster le seuil de décision entre 0.0 et 1.0
  - Visualisation dynamique de la **matrice de confusion** (Seaborn heatmap)

---

##  Architecture du Projet

```
scoring-bancaire/
│
├── app.py              # Script principal de l'application Streamlit
├── projet.ipynb        # Notebook d'analyse exploratoire et de développement du modèle
├── german.data         # Dataset source (German Credit Data - UCI)
└── README.md           # Documentation du projet
```

### Description des fichiers

| Fichier | Description |
|---|---|
| `app.py` | Application Streamlit principale : chargement des données, pipeline ML, interface utilisateur |
| `projet.ipynb` | Notebook Jupyter pour l'exploration, la visualisation et le prototypage du modèle |
| `german.data` | Dataset brut contenant 1000 observations et 20 variables sur des clients bancaires |

---

##  Prérequis

- Python **3.8+**
- pip

### Installation des dépendances

```bash
pip install pandas seaborn matplotlib scikit-learn streamlit
```

Ou via un fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```

**Contenu du `requirements.txt` :**

```
pandas
seaborn
matplotlib
scikit-learn
streamlit
```

---

##  Lancement de l'Application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse :
```
http://localhost:8501
```

---

## Pipeline Machine Learning

```
Chargement des données (UCI - german.data)
        │
        ▼
Nettoyage & Encodage (get_dummies / One-Hot Encoding)
        │
        ▼
Division Train / Test (80% / 20%, random_state=42)
        │
        ▼
Normalisation (StandardScaler)
        │
        ▼
Entraînement (LogisticRegression)
        │
        ▼
Prédiction des probabilités (predict_proba)
        │
        ▼
Application du seuil de décision (slider)
        │
        ▼
Évaluation (Matrice de Confusion)
```

---

## Données Utilisées

**Source :** [UCI Machine Learning Repository - Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

- **1000 observations**, **20 variables** (statut bancaire, durée du crédit, historique de crédit, montant, emploi, etc.)
- **Variable cible** : `target`
  - `1` → Client **solvable** (bon risque)
  - `0` → Client **non solvable** (mauvais risque)

---

## Interface Utilisateur

### Panneau principal
- **Checkbox** : Afficher ou masquer les 10 premières lignes du dataset
- **Matrice de Confusion** : Mise à jour dynamique selon le seuil sélectionné

### Barre latérale (Sidebar)
- **Slider — Seuil de décision** : Ajustable de `0.0` à `1.0` par pas de `0.05`
  - Un seuil **bas** → plus de clients classés comme solvables (moins de faux négatifs, plus de faux positifs)
  - Un seuil **élevé** → plus de rigueur dans l'acceptation (moins de faux positifs, plus de faux négatifs)

---

## Exemple de Résultat

Avec un seuil de décision à `0.5` (valeur par défaut), la matrice de confusion affiche :

|  | Prédit : Non solvable | Prédit : Solvable |
|---|---|---|
| **Réel : Non solvable** | Vrais Négatifs (TN) | Faux Positifs (FP) |
| **Réel : Solvable** | Faux Négatifs (FN) | Vrais Positifs (TP) |

> **Astuce** : Dans un contexte bancaire, les **faux négatifs** (clients solvables refusés) et les **faux positifs** (clients insolvables acceptés) n'ont pas le même coût. Ajustez le seuil selon la politique de risque souhaitée.

---

## Améliorations Possibles

- [ ] Ajout d'autres modèles (Random Forest, XGBoost, SVM) pour comparaison
- [ ] Affichage de métriques supplémentaires : AUC-ROC, précision, rappel, F1-score
- [ ] Courbe ROC interactive
- [ ] Formulaire de saisie manuelle pour prédire le profil d'un nouveau client
- [ ] Export des résultats en PDF ou CSV
- [ ] Déploiement sur Streamlit Cloud ou Hugging Face Spaces

---

## Auteur

Développé dans le cadre d'un projet de Data Science appliqué à la finance et au scoring de crédit.

---

## Licence

Ce projet est distribué à des fins éducatives. Le dataset utilisé est public et disponible sur le [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).