# Suivi et prédiction du taux de traitement par profil

Application Streamlit — suivi du taux de sollicitation (appels + emails) par
profil de compte, avec détection d'anomalie et prévision.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Testé avec **Python 3.12**. Les versions du `requirements.txt` sont volontairement
figées (une expérience passée avec des contraintes de version trop lâches avait
provoqué un segfault) : ajustez si besoin, mais testez en conséquence.

## Lancement

```bash
streamlit run main.py
```

L'application s'ouvre par défaut sur des **données de démonstration synthétiques**
(aucune donnée réelle requise) : ~11 000 comptes répartis en 5 profils (dont
« Autres »), sur 18 mois, avec deux canaux (appels/emails), une part de
traitements sans compte identifié, et une rupture de tendance injectée sur le
canal appel du profil « Digital ».

## Basculer sur vos propres données

Dans la barre latérale, choisissez « Importer mes fichiers CSV » puis déposez :

- **Traitements** : `traitement_id, account_id, date_contact, type_traitement` (appel/email)
  — `date_contact` au format **JJ/MM/AAAA HH:mm:ss** (ex. `05/03/2026 14:30:00`)
- **Portefeuille** : `account_id, profil_partenaire, relation_contractuelle`

Le portefeuille est **statique** (pas d'historique) : chaque compte a un seul
profil et une seule relation contractuelle, fixes dans le temps — pas de
`date_snapshot`, pas de jointure "as of", juste une fusion directe sur `account_id`.

Les noms de colonnes réels peuvent varier (accents, espaces, casse — ex. « Id_compte »,
« Date de création / de contact ») : une table d'alias (voir `data/loader.py::_map_columns`)
les reconnaît automatiquement, avec un repli par mots-clés en dernier recours. Si
l'import échoue, le message d'erreur liste les colonnes trouvées dans votre
fichier pour diagnostiquer l'écart. Les catégories (profil, relation
contractuelle) sont aussi normalisées automatiquement (casse/espaces) pour
qu'une même catégorie mal saisie ("Premium", "premium ", "PREMIUM") ne soit
pas fragmentée en plusieurs sous-catégories.

**account_id manquant** : quand `account_id` est vide dans Traitements.csv, le
profil ET la relation contractuelle sont automatiquement mis à "Autres"
(résolution dans `data/aggregator.py::resolve_profile_and_relation`), sans
fusion possible. Pareil si le compte n'apparaît pas du tout dans le portefeuille.

## Architecture

```
main.py                 point d'entrée Streamlit (orchestration + 4 onglets)
config.py                constantes, schémas, alias de colonnes
data/
  loader.py               génération synthétique + import CSV (schéma réel)
  aggregator.py            résolution profil/relation (fusion directe), filtres canal/relation, agrégation
metrics/
  call_rate.py             taux de traitement (2 définitions)
  lift.py                  indice de représentation (lift) + IC (binomial ou bootstrap)
  anomaly.py               écart à l'historique propre du profil + rupture (CUSUM)
forecasting/
  registry.py               MODEL_REGISTRY (8 modèles, dont 2 Binomiale négative)
  backtest.py                validation par repli temporel (walk-forward), comparatif des 8 modèles
ui/
  controls.py               widgets de la barre latérale (dont canal, relation)
  charts.py                 graphiques Plotly
```

## Validation des modèles

L'onglet Prévision propose un **comparatif des 8 modèles par validation
glissante (walk-forward)** : chaque modèle est ré-entraîné sur l'historique
disponible à plusieurs dates de coupure, puis évalué sur les périodes
suivantes réellement observées (erreur hors-échantillon, pas un simple
ajustement). Le classement inclut MAE, RMSE, biais, et surtout la
**couverture de l'intervalle de confiance** (cible ~95 % ; une couverture
nettement inférieure indique un modèle trop confiant — typiquement les
variantes Poisson sur une série sur-dispersée). Calcul à la demande (bouton),
peut prendre 1-2 minutes selon les modèles inclus (Prophet/PyMC sont plus lents).

## Points d'attention méthodologiques

- **"Autres" est ambigu** : ce panier mélange de vrais comptes classés "Autres"
  dans le portefeuille ET des traitements sans compte identifiable (qui ne
  comptent dans aucun dénominateur). Un indicateur de qualité de donnée séparé
  (onglet Vue synthétique) rend cette part visible plutôt que de la masquer.
- **Canal et relation contractuelle sont des filtres indépendants** du profil,
  pas des axes croisés systématiquement — pour ne pas diviser des volumes déjà
  surveillés par le garde-fou de fiabilité statistique.
- **Distribution non-Poisson** : mélanger appels et emails (ou filtrer sur un
  canal marqué par des pics ponctuels) produit typiquement de la sur-dispersion
  (variance > moyenne). Un diagnostic (ratio variance/moyenne) s'affiche dans
  l'onglet Prévision ; au-delà du seuil, préférez les variantes **Binomiale
  négative** du menu de modèles aux variantes Poisson, qui sous-estimeraient
  l'incertitude.
- **Alertes** : comparaison toujours par rapport à l'historique PROPRE du
  profil, jamais par rapport aux autres profils.
- **Prévision** : les modèles à tendance extrapolent sur toute la période
  d'analyse choisie. En cas de rupture nette (onglet Alertes), restreignez la
  période à l'après-rupture avant de lancer la prévision.
- La dernière période (jour/semaine/mois en cours, incomplète) est
  automatiquement exclue de l'analyse pour ne pas biaiser taux, lift et
  prévisions par un simple effet de troncature.
