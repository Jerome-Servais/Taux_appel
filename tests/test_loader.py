import io

import numpy as np
import pandas as pd
import pytest

from data.aggregator import aggregate_treatments_by_motif, build_master_table, resolve_profile_and_relation
from data.loader import load_demo_data, load_treatments_from_csv
from data.motif_taxonomy import validate_pair


def test_import_keeps_motif_columns():
    csv = io.StringIO(
        "Id,Id_compte,Date de contact,Canal,Motif,Sous-motif\n"
        "1,10,05/03/2026 14:30:00,Appel,Facturation, Remboursement \n"
        "2,,06/03/2026 09:00:00,Email,,\n"
    )
    df = load_treatments_from_csv(csv)
    assert df.loc[0, "motif"] == "Facturation"
    assert df.loc[0, "sous_motif"] == "Remboursement"
    assert pd.isna(df.loc[1, "motif"])
    assert df.loc[0, "date_contact"] == pd.Timestamp("2026-03-05 14:30:00")


def test_import_without_motif_still_works():
    csv = io.StringIO("traitement_id,account_id,date_contact,type_traitement\n1,10,05/03/2026 14:30:00,appel\n")
    df = load_treatments_from_csv(csv)
    assert df["motif"].isna().all()


@pytest.mark.parametrize("motif, sous_motif, valid", [
    ("Facturation", "Remboursement / avoir", True),
    ("Facturation", "Retard de collecte", False),
    (None, None, True),
    (None, "Retard de collecte", False),
])
def test_taxonomy_hierarchy(motif, sous_motif, valid):
    assert validate_pair(motif, sous_motif) is valid


def test_demo_motif_aggregation_conserves_volume():
    treatments, portfolio = load_demo_data()
    resolved = resolve_profile_and_relation(treatments, portfolio)
    table = aggregate_treatments_by_motif(resolved, "MS")
    assert table["n_traitements"].sum() == len(resolved)
    assert np.isclose(resolved["motif"].isna().mean(), 0.2, atol=0.05)


def test_master_table_drops_partial_first_and_last_periods():
    days = pd.date_range("2026-01-07 09:00", "2026-02-11 09:00", freq="D")  # mercredi -> mercredi
    treatments = pd.DataFrame({
        "traitement_id": range(len(days)), "account_id": 1, "date_contact": days,
        "profil_partenaire": "Standard",
    })
    portfolio = pd.DataFrame({"account_id": [1], "profil_partenaire": ["Standard"]})

    master = build_master_table(treatments, portfolio, "W-MON")

    assert master["period"].min() == pd.Timestamp("2026-01-12")
    assert master["period"].max() == pd.Timestamp("2026-02-02")
    assert (master["n_traitements"] == 7).all()
