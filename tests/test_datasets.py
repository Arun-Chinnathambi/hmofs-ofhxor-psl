"""Loader tests using tiny mock files in each dataset's native layout."""
import numpy as np
import pandas as pd
import pytest

from hmofs_ofhxor_psl.datasets import (load_cinc, load_ecu_ioht, load_mimic3,
                                       load_uci_heart, load_wustl_ehms)


def test_uci_raw_cleveland_no_header(tmp_path):
    p = tmp_path / "processed.cleveland.data"
    p.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
                 "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n"
                 "37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,?,3.0,0\n")
    X, y = load_uci_heart(p)
    assert list(y) == [0, 1, 0] and X.shape == (3, 13)
    assert not X.isna().any().any()                  # '?' imputed


def test_uci_kaggle_header(tmp_path):
    p = tmp_path / "heart.csv"
    pd.DataFrame({"age": [50, 60], "chol": [200, 250], "target": [0, 1]}).to_csv(p, index=False)
    X, y = load_uci_heart(p)
    assert list(y) == [0, 1] and "target" not in X.columns


def test_cinc2012(tmp_path):
    d = tmp_path / "set-a"
    d.mkdir()
    for rid in (1, 2):
        (d / f"{rid}.txt").write_text(
            "Time,Parameter,Value\n00:00,RecordID,%d\n00:00,Age,%d\n00:00,Gender,0\n"
            "00:00,Height,-1\n00:00,ICUType,2\n00:00,Weight,-1\n"
            "00:07,HR,73\n00:37,HR,%d\n01:07,Temp,36.8\n" % (rid, 60 + rid, 80 + rid))
    out = tmp_path / "Outcomes-a.txt"
    out.write_text("RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
                   "1,10,2,5,-1,0\n2,20,6,9,3,1\n")
    X, y = load_cinc(str(d), str(out))
    assert list(y) == [0, 1]
    assert "HR_mean" in X.columns and "HR_max" in X.columns
    assert "Age" in X.columns and not X.isna().any().any()


def test_cinc2019(tmp_path):
    d = tmp_path / "training"
    d.mkdir()
    for i, lab in enumerate([0, 1]):
        pd.DataFrame({"HR": [80, 90, 95], "O2Sat": [98, 97, 96], "Temp": [37, 37.5, 38],
                      "Age": [60] * 3, "ICULOS": [1, 2, 3],
                      "SepsisLabel": [0, 0, lab]}).to_csv(d / f"p{i}.psv", sep="|", index=False)
    X, y = load_cinc(str(d))
    assert list(y) == [0, 1] and "SepsisLabel" not in X.columns
    assert X["ICULOS"].iloc[0] == 3


def test_mimic3_extract_and_cache(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for hadm in range(100, 110):
        for item in (211, 220045, 51, 8368, 618, 646, 678):      # includes a Fahrenheit temp
            val = {678: 98.6}.get(item, rng.uniform(60, 100))
            rows.append((hadm, item, val))
    rows.append((None, 211, 80.0))                               # missing HADM_ID
    pd.DataFrame(rows, columns=["HADM_ID", "ITEMID", "VALUENUM"]).assign(
        ROW_ID=range(len(rows))).to_csv(tmp_path / "CHARTEVENTS.csv", index=False)
    pd.DataFrame({"HADM_ID": range(100, 110),
                  "HOSPITAL_EXPIRE_FLAG": [0, 1] * 5}).to_csv(tmp_path / "ADMISSIONS.csv",
                                                              index=False)
    cache = str(tmp_path / "cache" / "v.csv")
    X, y = load_mimic3(str(tmp_path), cache_csv=cache)
    assert len(X) == len(y) == 10 and set(y) == {0, 1}
    assert {"HR", "SBP", "DBP", "RR", "SpO2", "Temp"} <= set(X.columns)
    assert X["Temp"].between(36, 38).all()                       # 98.6F -> 37C
    X2, y2 = load_mimic3(str(tmp_path), cache_csv=cache)         # reads cache
    assert X2.shape == X.shape and list(y2) == list(y)


def test_wustl_drops_identifiers_and_strips_names(tmp_path):
    p = tmp_path / "wustl.csv"
    pd.DataFrame({"SrcAddr": ["a", "b", "a"], "DstAddr": ["c", "c", "d"],
                  "Sport": [1, 2, 3], "Dport": [80, 80, 80], "SrcBytes": [10, 20, 30],
                  "SYS ": [120, 130, 125], "Temp": [36.5, 37, 36.9],
                  "Attack Category": ["normal", "Spoofing", "normal"],
                  "Label": [0, 1, 0]}).to_csv(p, index=False)
    X, y = load_wustl_ehms(p)
    assert list(y) == [0, 1, 0]
    assert "SrcAddr" not in X.columns and "Sport" not in X.columns
    assert "SYS" in X.columns                                    # trailing space stripped
    assert "Attack Category" not in X.columns and "Label" not in X.columns


def test_ecu_ioht(tmp_path):
    p = tmp_path / "ecu.csv"
    pd.DataFrame({"Source": ["1.1.1.1", "2.2.2.2", "1.1.1.1"],
                  "Destination": ["3.3.3.3"] * 3, "Protocol": ["ARP", "TCP", "ARP"],
                  "Length": [42, 60, 42], "Info": ["Who has 10.0.0.1?", "SYN", "ARP reply"],
                  "Type": ["Normal", "DoS", "ARP Spoofing"]}).to_csv(p, index=False)
    X, y = load_ecu_ioht(p)
    assert list(y) == [0, 1, 1]
    assert "Info_len" in X.columns and "Source" not in X.columns
    X2, _ = load_ecu_ioht(p, keep_addresses=True)
    assert "Source" in X2.columns


def test_missing_label_raises(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_wustl_ehms(p)
