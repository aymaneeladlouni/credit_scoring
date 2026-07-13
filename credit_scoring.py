"""
MODELE DE CREDIT SCORING
Auteur : Aymane El Adlouni

Ce programme estime la probabilite qu'un emprunteur fasse defaut sur son
credit, c'est-a-dire qu'il ne rembourse pas. C'est exactement ce que font
les banques pour decider d'accorder ou non un pret.

Le modele apprend a partir d'un historique de clients dont on connait
deja l'issue (ont-ils rembourse ou non), puis il predit le risque pour
de nouveaux clients.

Etapes du modele :
    1. Charger et preparer les donnees des clients
    2. Separer les donnees en un jeu d'entrainement et un jeu de test
    3. Entrainer une regression logistique (le standard bancaire)
    4. Predire la probabilite de defaut pour chaque client du test
    5. Mesurer la performance du modele
    6. Identifier les variables les plus importantes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


def generer_donnees(n_clients=1000, graine=42):
    """
    Genere un jeu de donnees realiste de clients demandant un credit.

    Dans un vrai projet, ces donnees viendraient d'un fichier CSV de la
    banque. Ici on les simule pour que le modele soit reproductible par
    n'importe qui, sans donnees confidentielles.

    Chaque client est decrit par plusieurs variables (revenu, age, montant
    du pret, etc.) et par une cible : a-t-il fait defaut (1) ou non (0).
    """
    np.random.seed(graine)

    age = np.random.randint(21, 65, n_clients)
    revenu_mensuel = np.random.normal(8000, 3000, n_clients).clip(2500, 25000)
    montant_pret = np.random.normal(120000, 60000, n_clients).clip(10000, 400000)
    duree_pret_mois = np.random.choice([12, 24, 36, 48, 60, 84], n_clients)
    anciennete_emploi = np.random.randint(0, 30, n_clients)
    nb_credits_en_cours = np.random.poisson(1.2, n_clients)

    # Le taux d'endettement : part du revenu annuel absorbee par le pret.
    # C'est l'un des indicateurs de risque les plus surveilles en banque.
    mensualite = montant_pret / duree_pret_mois
    taux_endettement = (mensualite / revenu_mensuel).clip(0, 2)

    # Construction d'un "score de risque" latent : plus il est eleve,
    # plus le client a de chances de faire defaut. On combine les facteurs
    # de risque avec des poids realistes, puis on ajoute du hasard.
    risque = (
        1.8 * taux_endettement
        - 0.00004 * revenu_mensuel
        + 0.12 * nb_credits_en_cours
        - 0.03 * anciennete_emploi
        + 0.000002 * montant_pret
        + np.random.normal(0, 0.4, n_clients)
    )

    # On transforme ce risque en probabilite (entre 0 et 1) via la fonction
    # logistique, puis on tire le defaut au hasard selon cette probabilite.
    proba = 1 / (1 + np.exp(-(risque - 0.9)))
    defaut = np.random.binomial(1, proba)

    donnees = pd.DataFrame({
        "age": age,
        "revenu_mensuel": revenu_mensuel.round(0),
        "montant_pret": montant_pret.round(0),
        "duree_pret_mois": duree_pret_mois,
        "anciennete_emploi": anciennete_emploi,
        "nb_credits_en_cours": nb_credits_en_cours,
        "taux_endettement": taux_endettement.round(3),
        "defaut": defaut,
    })

    return donnees


def explorer_donnees(donnees):
    """Affiche un apercu des donnees et du taux de defaut global."""
    print("Apercu des donnees clients")
    print(donnees.head(8).to_string(index=False))
    print()

    taux_defaut = donnees["defaut"].mean() * 100
    print(f"Nombre de clients        : {len(donnees)}")
    print(f"Taux de defaut global    : {taux_defaut:.1f}%")
    print(f"Clients sains            : {(donnees['defaut'] == 0).sum()}")
    print(f"Clients en defaut        : {(donnees['defaut'] == 1).sum()}")
    print()


def entrainer_modele(donnees):
    """
    Entraine la regression logistique sur les donnees clients.

    On separe d'abord les variables explicatives (X) de la cible (y),
    puis on divise en jeu d'entrainement (pour apprendre) et jeu de test
    (pour evaluer honnetement le modele sur des clients jamais vus).
    """
    variables = [
        "age",
        "revenu_mensuel",
        "montant_pret",
        "duree_pret_mois",
        "anciennete_emploi",
        "nb_credits_en_cours",
        "taux_endettement",
    ]

    X = donnees[variables]
    y = donnees["defaut"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # On standardise les variables (meme echelle) pour que la regression
    # traite chaque facteur equitablement, quel que soit son ordre de grandeur.
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    modele = LogisticRegression(max_iter=1000)
    modele.fit(X_train_std, y_train)

    return modele, scaler, X_test_std, y_test, variables


def evaluer_modele(modele, X_test_std, y_test):
    """
    Mesure la qualite des predictions du modele sur le jeu de test.

    On regarde la matrice de confusion, le rapport de classification et
    surtout l'AUC (aire sous la courbe ROC), la mesure de reference pour
    juger un modele de scoring : plus elle est proche de 1, mieux le modele
    distingue les bons des mauvais payeurs.
    """
    predictions = modele.predict(X_test_std)
    probabilites = modele.predict_proba(X_test_std)[:, 1]

    print("Matrice de confusion")
    mc = confusion_matrix(y_test, predictions)
    print(f"                 Predit sain   Predit defaut")
    print(f"Reel sain        {mc[0][0]:>11}   {mc[0][1]:>13}")
    print(f"Reel defaut      {mc[1][0]:>11}   {mc[1][1]:>13}")
    print()

    print("Rapport de classification")
    print(classification_report(y_test, predictions,
                                target_names=["Sain", "Defaut"]))

    auc = roc_auc_score(y_test, probabilites)
    print(f"AUC (aire sous la courbe ROC) : {auc:.3f}")
    print()

    return probabilites, auc


def importance_variables(modele, variables):
    """
    Classe les variables selon leur poids dans la decision du modele.

    Un coefficient positif augmente le risque de defaut, un coefficient
    negatif le diminue. C'est ce qui rend la regression logistique
    appreciee en banque : le modele est interpretable, on sait pourquoi
    il accorde ou refuse un credit.
    """
    coefficients = modele.coef_[0]
    importance = pd.DataFrame({
        "variable": variables,
        "coefficient": coefficients,
    }).sort_values("coefficient", key=abs, ascending=False)

    print("Importance des variables (poids dans la decision)")
    print(importance.to_string(index=False))
    print()

    return importance


def tracer_resultats(y_test, probabilites, importance):
    """Trace la courbe ROC et l'importance des variables cote a cote."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_test, probabilites)
    auc = roc_auc_score(y_test, probabilites)
    ax1.plot(fpr, tpr, color="#4C72B0", linewidth=2,
             label=f"Modele (AUC = {auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--",
             label="Hasard (AUC = 0.5)")
    ax1.set_xlabel("Taux de faux positifs")
    ax1.set_ylabel("Taux de vrais positifs")
    ax1.set_title("Courbe ROC")
    ax1.legend()
    ax1.grid(alpha=0.3)

    couleurs = ["#C44E52" if c > 0 else "#55A868"
                for c in importance["coefficient"]]
    ax2.barh(importance["variable"], importance["coefficient"], color=couleurs)
    ax2.set_xlabel("Coefficient (positif = plus de risque)")
    ax2.set_title("Importance des variables")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("credit_scoring_resultats.png", dpi=120)
    print("Graphique enregistre : credit_scoring_resultats.png")


if __name__ == "__main__":
    donnees = generer_donnees()
    explorer_donnees(donnees)

    modele, scaler, X_test_std, y_test, variables = entrainer_modele(donnees)
    probabilites, auc = evaluer_modele(modele, X_test_std, y_test)
    importance = importance_variables(modele, variables)
    tracer_resultats(y_test, probabilites, importance)
