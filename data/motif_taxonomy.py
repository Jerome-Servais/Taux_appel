"""
data/motif_taxonomy.py — Taxonomie motif / sous-motif (hiérarchique).

Un sous_motif n'appartient qu'à un seul motif (relation N-à-1 stricte).

Motifs de contact de la démonstration, typiques d'un éco-organisme :
- Déclaration       : déclaration d'emballages, mise à jour des données,
                      déclaration rectificative
- Facturation       : demande de facture, contestation éco-contribution,
                      remboursement, acompte
- Point de reprise  : création, modification ou suppression d'un point de
                      reprise (PDR), question sur la collecte
- Réglementaire     : AGEC, REP, conformité, audit, nouvelle obligation
- Réclamation       : retard de collecte, qualité de service, litige,
                      réclamation diverse

Part de non-qualification : certains traitements arrivent sans motif connu
(ex. appels raccrochard avant identification du besoin, emails sans objet
structuré). Modélisé par la constante MISSING_MOTIF_RATE.

DICTIONNAIRE DE DONNÉES — colonnes ajoutées à la table des traitements
─────────────────────────────────────────────────────────────────────────
| Colonne     | Type   | Nullable | Description                          |
|-------------|--------|----------|--------------------------------------|
| motif       | str    | Oui      | Catégorie de 1er niveau du contact.  |
|             |        |          | Vaut None si non qualifié.           |
| sous_motif  | str    | Oui      | Catégorie de 2e niveau, enfant du    |
|             |        |          | motif. Vaut None si motif absent ou  |
|             |        |          | si le détail n'a pas été renseigné.  |
─────────────────────────────────────────────────────────────────────────

Contrainte d'intégrité garantie par la constante TAXONOMY : pour toute
ligne où sous_motif est non-null, motif est également non-null, et le
sous_motif appartient bien au motif déclaré.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Taxonomie canonique
# ---------------------------------------------------------------------------

# Structure : {motif: [(sous_motif, poids_relatif), ...]}
# Les poids sont normalisés automatiquement lors du tirage.
TAXONOMY: dict[str, list[tuple[str, float]]] = {
    "Déclaration": [
        ("Déclaration annuelle d'emballages", 0.45),
        ("Mise à jour des données déclarées", 0.30),
        ("Déclaration rectificative", 0.25),
    ],
    "Facturation": [
        ("Demande de facture / justificatif", 0.35),
        ("Contestation éco-contribution", 0.30),
        ("Remboursement / avoir", 0.20),
        ("Question acompte / calendrier", 0.15),
    ],
    "Point de reprise": [
        ("Création d'un nouveau PDR", 0.30),
        ("Modification d'un PDR existant", 0.25),
        ("Suppression d'un PDR", 0.15),
        ("Question sur la collecte", 0.30),
    ],
    "Réglementaire": [
        ("AGEC / REP — interprétation", 0.40),
        ("Nouvelle obligation déclarative", 0.30),
        ("Audit / conformité", 0.20),
        ("Autre question réglementaire", 0.10),
    ],
    "Réclamation": [
        ("Retard de collecte", 0.35),
        ("Qualité de service", 0.30),
        ("Litige facturation", 0.25),
        ("Réclamation diverse", 0.10),
    ],
}

# Poids inter-motifs (probabilité qu'un traitement qualifié appartienne à ce motif)
MOTIF_WEIGHTS: dict[str, float] = {
    "Déclaration": 0.30,
    "Facturation": 0.28,
    "Point de reprise": 0.18,
    "Réglementaire": 0.12,
    "Réclamation": 0.12,
}

# Taux de non-qualification (motif absent) par canal — par cohérence avec
# MISSING_ACCOUNT_RATE de data/loader.py : un email non authentifié est
# encore moins susceptible d'être qualifié en motif qu'un appel pris en charge
# par un agent.
MISSING_MOTIF_RATE: dict[str, float] = {
    "appel": 0.12,
    "email": 0.28,
}

# Taux supplémentaire où le motif est présent mais le sous-motif non renseigné
# (ex. agent qui ferme le dossier sans renseigner le détail)
MISSING_SOUS_MOTIF_RATE: float = 0.08

# Couple plat (motif, sous_motif) pour lookup rapide
MOTIF_LIST: list[str] = list(TAXONOMY.keys())

SOUS_MOTIF_PARENT: dict[str, str] = {
    sm: motif
    for motif, subs in TAXONOMY.items()
    for sm, _ in subs
}

SOUS_MOTIF_LIST: list[str] = list(SOUS_MOTIF_PARENT.keys())


def get_sous_motifs(motif: str) -> list[str]:
    """Retourne les sous-motifs d'un motif donné (liste vide si inconnu)."""
    return [sm for sm, _ in TAXONOMY.get(motif, [])]


def validate_pair(motif: str | None, sous_motif: str | None) -> bool:
    """Vérifie la cohérence hiérarchique d'une paire (motif, sous_motif)."""
    if motif is None:
        return sous_motif is None
    if motif not in TAXONOMY:
        return False
    if sous_motif is None:
        return True  # motif présent, sous-motif absent : autorisé
    return SOUS_MOTIF_PARENT.get(sous_motif) == motif
