# HMOFS-OFHXOR-PSL

Integrated system for **secure healthcare IoT monitoring**: Hybrid Multi-Objective
Feature Selection (HMOFS), Optimal Fire Hawk XOR encryption (OFHXOR) and a Probabilistic
Super Learner (PSL) for attack detection.

## Datasets and their roles

| Dataset | Stage | Label used | Loader |
|---|---|---|---|
| MIMIC-III Clinical Database | Encryption (telemetry / vitals) | hospital mortality | `datasets/mimic3.py` |
| PhysioNet / CinC (2012 or 2019) | Encryption | in-hospital death / sepsis | `datasets/physionet_cinc.py` |
| UCI Heart Disease | Encryption | heart disease (num > 0) | `datasets/uci_heart.py` |
| WUSTL-EHMS-2020 | Detection (+ its vitals as an encryption source) | attack | `datasets/wustl_ehms.py` |
| ECU-IoHT | Detection | attack (any non-normal type) | `datasets/ecu_ioht.py` |

**Important:** the three clinical datasets contain no attacks and no network traffic.
HMOFS on them selects features against *clinical outcomes*, and in simulation they act as
payloads protected by OFHXOR, paired with WUSTL traffic for the PSL's decision. The PSL is
trained and evaluated only on WUSTL-EHMS-2020 and ECU-IoHT (one detector per dataset,
since schemas differ). Report results accordingly.

## Architecture

| Stage | Block | Module |
|---|---|---|
| Encryption | Feature Selection (HMOFS), per clinical source | `hmofs.py` |
| Encryption | Optimal key generation (FHO), H(K) = -Σ p(k) log2 p(k) | `fho.py`, `keygen.py` |
| Encryption | OFHXOR (K\*, D1, D2) | `ofhxor.py` |
| Encryption | Entropy / correlation / avalanche metrics | `metrics.py` |
| Detection | PSL (α1, α2, α3), posterior fusion, θ = 0.45 | `psl.py`, `detector.py` |
| Detection layer | Risk level assignment + feedback loop | `detection_layer.py` |
| Storage | Verified, encrypted data (HMAC-SHA256) | `storage.py` |
| Orchestration | End-to-end flow | `pipeline.py`, `main.py` |

```
MIMIC-III ─┐
CinC ──────┼─> HMOFS (per source) -> FHO key K* -> OFHXOR ──────────┐
UCI Heart ─┤                                                        v
WUSTL vitals┘                                                 Secure storage
WUSTL-EHMS ─┐                                                (HMAC-verified)
ECU-IoHT ───┴─> HMOFS -> PSL (θ=0.45) -> Risk level ───────────────^
                              ^              |
                              +-- feedback --+
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                      # all-synthetic demo (no data needed)

python main.py \
  --mimic  /path/to/mimiciii/           # folder with CHARTEVENTS + ADMISSIONS (or extracted CSV)
  --cinc   data/set-a --cinc-outcomes data/Outcomes-a.txt   # CinC 2012
  # --cinc data/training_setA           # CinC 2019 (.psv files, no outcomes file needed)
  --uci    data/processed.cleveland.data \
  --wustl  data/wustl-ehms-2020.csv \
  --ecu    data/ECU-IoHT.csv
```

Any dataset you don't pass is replaced by a synthetic stand-in, so you can add them one
at a time. Put downloaded data in `data/` (git-ignored).

### Data handling notes
- **MIMIC-III and PhysioNet CinC require credentialed access / data use agreements.**
  Never commit them, derived per-patient files, or share them with hosted services.
  The MIMIC vitals extraction is cached to `data/mimic3_vitals_cache.csv` (git-ignored).
- MIMIC vital-sign ITEMIDs in `mimic3.py` cover CareVue and MetaVision; verify them against
  `D_ITEMS.csv` for your release. Time series are reduced to one mean per admission.
- WUSTL identifiers (`SrcAddr`, `DstAddr`, `Sport`, `Dport`, MACs) are dropped by default
  because host identity leaks the label. ECU-IoHT `Source`/`Destination` are dropped unless
  `--ecu-keep-addresses` is passed; `Info` becomes `Info_len`.
- Check your CSV column names with `df.columns`; loaders strip whitespace (e.g. `"SYS "`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # includes loader tests on mock files in each dataset's native format
```

## Design notes and limitations

- **PSL:** α weights are learned on the simplex by minimising validation log-loss; an
  attack is flagged when the fused posterior is ≥ θ = 0.45.
- **Risk levels:** LOW < 0.20 ≤ MEDIUM < 0.45 ≤ HIGH < 0.75 ≤ CRITICAL (CRITICAL is blocked).
  Edit `RISK_LEVELS` in `config.py`.
- **OFHXOR:** the (K\*, D1, D2) block structure is an interpretation of the architecture
  diagram. A chained XOR stream gives weak diffusion (avalanche ≈ 0.1, ideal 0.5) and
  is malleable; integrity here comes from the HMAC in `storage.py`. For real patient data
  use an authenticated cipher such as AES-GCM.
- Detection is binary (normal vs. attack), not per attack type.
