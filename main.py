"""Entry point: runs the full HMOFS-OFHXOR-PSL pipeline on up to five datasets.

    python main.py                                     # all-synthetic demo
    python main.py --mimic /path/to/mimiciii/ --uci data/processed.cleveland.data \
                   --cinc data/set-a --cinc-outcomes data/Outcomes-a.txt \
                   --wustl data/wustl-ehms-2020.csv --ecu data/ECU-IoHT.csv
Any dataset not supplied is replaced by a synthetic stand-in.
"""
import argparse

import numpy as np

from hmofs_ofhxor_psl import HealthcareSecurityPipeline
from hmofs_ofhxor_psl.config import KEY_LENGTH, THETA
from hmofs_ofhxor_psl.datasets import (WUSTL_VITALS, load_cinc, load_ecu_ioht, load_mimic3,
                                       load_uci_heart, load_wustl_ehms, synth_clinical,
                                       synth_ecu_like, synth_wustl_like)

WUSTL, ECU = "WUSTL-EHMS-2020", "ECU-IoHT"
WUSTL_V = "WUSTL-EHMS-2020 (vitals)"


def parse():
    ap = argparse.ArgumentParser(description="HMOFS-OFHXOR-PSL secure healthcare IoT system")
    ap.add_argument("--mimic", help="MIMIC-III folder (CHARTEVENTS, ADMISSIONS) or extracted CSV")
    ap.add_argument("--cinc", help="CinC folder (.txt 2012 / .psv 2019) or pre-processed CSV")
    ap.add_argument("--cinc-outcomes", help="Outcomes-*.txt (CinC 2012 only)")
    ap.add_argument("--uci", help="UCI Heart Disease CSV / processed.cleveland.data")
    ap.add_argument("--wustl", help="WUSTL-EHMS-2020 CSV")
    ap.add_argument("--ecu", help="ECU-IoHT CSV")
    ap.add_argument("--ecu-keep-addresses", action="store_true")
    ap.add_argument("--stream", type=int, default=300)
    return ap.parse_args()


def pick(path, loader, synth, label):
    if path:
        return loader(path)
    print(f"[info] {label}: no path given -> synthetic stand-in")
    return synth()


def main():
    a = parse()
    clinical = {
        "MIMIC-III": pick(a.mimic, load_mimic3, lambda: synth_clinical("MIMIC-III"), "MIMIC-III"),
        "PhysioNet-CinC": pick(a.cinc, lambda p: load_cinc(p, a.cinc_outcomes),
                               lambda: synth_clinical("PhysioNet-CinC"), "PhysioNet-CinC"),
        "UCI-Heart": pick(a.uci, load_uci_heart, lambda: synth_clinical("UCI-Heart"), "UCI-Heart"),
    }
    wdf, wy = pick(a.wustl, load_wustl_ehms, synth_wustl_like, WUSTL)
    edf, ey = pick(a.ecu, lambda p: load_ecu_ioht(p, a.ecu_keep_addresses), synth_ecu_like, ECU)

    wnames = list(wdf.columns)
    vit = [c for c in WUSTL_VITALS if c in wnames] or [c for c in wnames[-7:]]
    sources = dict(clinical)
    sources[WUSTL_V] = (wdf[vit], wy)          # WUSTL also carries patient vitals

    print("\n=== DATASETS ===")
    for n, (d, y) in {**clinical, WUSTL: (wdf, wy), ECU: (edf, ey)}.items():
        print(f"  {n:16s} {d.shape[0]:>7d} rows x {d.shape[1]:>3d} features | positives {y.mean():.1%}")

    pipe = HealthcareSecurityPipeline(key_len=KEY_LENGTH)

    print("\n=== ENCRYPTION PROCESS: HMOFS (clinical labels) -> FHO key -> OFHXOR ===")
    pipe.build_encryption_stage(sources)
    print(f"  optimal key K*: {pipe.key.hex()}")
    print(f"  H(K*) = {pipe.key_entropy:.3f} bits (max {np.log2(pipe.key_len):.3f})")
    print(f"  {'source':26s} {'features':>9s} {'entropy':>8s} {'|corr|':>7s} {'avalanche':>10s}")
    for n, k, tot, ent, corr, av in pipe.evaluate_encryption():
        print(f"  {n:26s} {k:>4d}/{tot:<4d} {ent:8.3f} {corr:7.3f} {av:10.3f}")
    print("  (avalanche ideal ~0.5; stream-XOR schemes score lower - see README)")

    print(f"\n=== ATTACK DETECTION PROCESS: HMOFS -> PSL (theta = {THETA}) ===")
    pipe.build_detection_stage({WUSTL: (wdf.values, wy, wnames),
                                ECU: (edf.values, ey, list(edf.columns))})
    for n, acc, f1, auc, al, k, tot in pipe.evaluate():
        print(f"  {n:16s} features {k:2d}/{tot:2d} | acc {acc:.4f} F1 {f1:.4f} AUC {auc:.4f}"
              f" | alpha = [{al[0]:.2f}, {al[1]:.2f}, {al[2]:.2f}]")

    print("\n=== DETECTION LAYER + SECURE STORAGE (simulation) ===")
    print("  clinical sets carry no network traffic: each session pairs a clinical payload")
    print("  with WUSTL traffic features, which the PSL judges.")
    Xte, yte = pipe.test_sets[WUSTL]
    vidx = [wnames.index(c) for c in vit]
    pools = {n: d.values.astype(float) for n, (d, _) in clinical.items()}
    pools[WUSTL_V] = Xte[:, vidx]
    names, stats, stored = list(sources), {}, {}
    for i in range(min(a.stream, len(Xte))):
        src = names[i % len(names)]
        row = pools[src][i % len(pools[src])]
        r = pipe.process(WUSTL, Xte[i], f"s{i}", src, row, yte[i])
        stats[r["risk"]] = stats.get(r["risk"], 0) + 1
        if r["action"] != "BLOCK":
            stored.setdefault(src, (r["session"], row[sources[src][0].columns.get_indexer(
                pipe.sources[src]["cols"])]))
        if i < 6:
            print(f"  {r['session']:5s} {src:26s} p={r['p']:.3f} {r['risk']:8s} {r['action']}")
    print(f"  risk distribution : {stats}")
    print(f"  feedback retrains : {pipe.layer.retrain_count}")

    for src, (sid, expected) in stored.items():
        vals, risk, _ = pipe.read_back(sid)
        ok = np.allclose(vals, expected.astype(np.float32))
        print(f"  verify {sid:6s} [{src}] decrypt matches original: {ok}")
    sid = next(iter(pipe.storage.vault))
    blob, r_, t_ = pipe.storage.vault[sid]
    pipe.storage.vault[sid] = (blob[:-1] + bytes([blob[-1] ^ 1]), r_, t_)   # tamper
    try:
        pipe.storage.get(sid)
    except ValueError as e:
        print(f"  tamper test       : detected -> {e}")


if __name__ == "__main__":
    main()
