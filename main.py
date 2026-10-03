"""Entry point: runs the full HMOFS-OFHXOR-PSL pipeline.

    python main.py                                            # synthetic demo
    python main.py --wustl wustl-ehms-2020.csv --ecu ECU-IoHT.csv
"""
import argparse

import numpy as np
import pandas as pd

from hmofs_ofhxor_psl import HealthcareSecurityPipeline
from hmofs_ofhxor_psl.config import KEY_LENGTH, THETA, VITALS
from hmofs_ofhxor_psl.data import load_tabular, synth_ecu_like, synth_wustl_like


def main():
    ap = argparse.ArgumentParser(description="HMOFS-OFHXOR-PSL secure healthcare IoT system")
    ap.add_argument("--wustl", help="path to WUSTL-EHMS-2020 CSV")
    ap.add_argument("--ecu", help="path to ECU-IoHT CSV")
    ap.add_argument("--stream", type=int, default=300, help="records to stream (default 300)")
    args = ap.parse_args()

    # ---- datasets
    if args.wustl:
        Xw, yw, nw, dfw = load_tabular(args.wustl)
        tele = dfw[[c for c in VITALS if c in dfw.columns]].apply(
            pd.to_numeric, errors="coerce").fillna(0)
    else:
        dfw, yw = synth_wustl_like()
        Xw, nw, tele = dfw.values, list(dfw.columns), dfw[VITALS]
        print("[info] --wustl not given -> using synthetic WUSTL-like data")
    if args.ecu:
        Xe, ye, ne, _ = load_tabular(args.ecu)
    else:
        dfe, ye = synth_ecu_like()
        Xe, ne = dfe.values, list(dfe.columns)
        print("[info] --ecu not given -> using synthetic ECU-IoHT-like data")

    pipe = HealthcareSecurityPipeline(key_len=KEY_LENGTH)

    print("\n=== ENCRYPTION PROCESS: HMOFS -> FHO key -> OFHXOR ===")
    pipe.build_encryption_stage(tele, yw)
    print(f"  telemetry features kept by HMOFS : {pipe.tele_cols}")
    print(f"  optimal key K*                   : {pipe.key.hex()}")
    print(f"  H(K*) = {pipe.key_entropy:.3f} bits (max {np.log2(pipe.key_len):.3f})")
    msg = b"vitals:36.9,97,72,118,79,73,16" * 3
    assert pipe.cipher.decrypt(pipe.cipher.encrypt(msg)) == msg
    print("  OFHXOR encrypt/decrypt round-trip OK")

    print(f"\n=== ATTACK DETECTION PROCESS: HMOFS -> PSL (theta = {THETA}) ===")
    pipe.build_detection_stage({"WUSTL-EHMS-2020": (Xw, yw, nw), "ECU-IoHT": (Xe, ye, ne)})
    for name, acc, f1, auc, alpha, k, total in pipe.evaluate():
        print(f"  {name:16s} features {k:2d}/{total:2d} | acc {acc:.4f} F1 {f1:.4f} "
              f"AUC {auc:.4f} | alpha = [{alpha[0]:.2f}, {alpha[1]:.2f}, {alpha[2]:.2f}]")

    print("\n=== DETECTION LAYER + SECURE STORAGE (streaming demo) ===")
    Xte, yte = pipe.test_sets["WUSTL-EHMS-2020"]
    tele_idx = [nw.index(c) for c in VITALS if c in nw]
    stats = {}
    for i in range(min(args.stream, len(Xte))):
        r = pipe.process("WUSTL-EHMS-2020", Xte[i], Xte[i][tele_idx], f"s{i}", yte[i])
        stats[r["risk"]] = stats.get(r["risk"], 0) + 1
        if i < 6:
            print(f"  {r['session']:4s} p={r['p']:.3f} risk={r['risk']:8s} action={r['action']}")
    print(f"  risk distribution : {stats}")
    print(f"  feedback retrains : {pipe.layer.retrain_count}")

    sid = next(iter(pipe.storage.vault))
    vals, risk = pipe.read_back(sid)
    print(f"  decrypted {sid} ({risk}): {np.round(vals, 2)}")
    blob, r_, t_ = pipe.storage.vault[sid]
    pipe.storage.vault[sid] = (blob[:-1] + bytes([blob[-1] ^ 1]), r_, t_)   # tamper
    try:
        pipe.storage.get(sid)
    except ValueError as e:
        print(f"  tamper test       : detected -> {e}")


if __name__ == "__main__":
    main()
