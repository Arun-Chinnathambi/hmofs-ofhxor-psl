# HMOFS-OFHXOR-PSL

Integrated system for **secure healthcare IoT monitoring**: Hybrid Multi-Objective
Feature Selection (HMOFS), Optimal Fire Hawk XOR encryption (OFHXOR) and a Probabilistic
Super Learner (PSL) for attack detection.

## Architecture

| Stage | Block | Module |
|---|---|---|
| Encryption | Feature Selection (HMOFS) | `hmofs_ofhxor_psl/hmofs.py` |
| Encryption | Optimal key generation (FHO), H(K) = -Σ p(k) log2 p(k) | `fho.py`, `keygen.py` |
| Encryption | OFHXOR (K\*, D1, D2) | `ofhxor.py` |
| Detection | WUSTL-EHMS-2020 / ECU-IoHT loading | `data.py` |
| Detection | PSL (α1, α2, α3), posterior fusion, θ = 0.45 | `psl.py`, `detector.py` |
| Detection layer | Risk level assignment + feedback loop | `detection_layer.py` |
| Storage | Verified, encrypted data (HMAC-SHA256) | `storage.py` |
| Orchestration | End-to-end flow | `pipeline.py`, `main.py` |

```
Telemetry -> HMOFS -> FHO key K* -> OFHXOR --------------------+
                                                                v
WUSTL-EHMS / ECU-IoHT -> HMOFS -> PSL (θ=0.45) -> Risk level -> Secure storage
                                       ^             |
                                       +-- feedback -+
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                      # synthetic demo
python main.py --wustl data/wustl-ehms-2020.csv --ecu data/ECU-IoHT.csv
```

Datasets are not included; download WUSTL-EHMS-2020 and ECU-IoHT from their original
sources into `data/` (git-ignored). The loader auto-detects the label column
(`Label`, `Attack Category`, `Type`) and treats any non-"normal" value as an attack.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Design notes and limitations

- **PSL:** α weights are learned on the simplex by minimising validation log-loss; an
  attack is flagged when the fused posterior is ≥ θ = 0.45.
- **Risk levels:** LOW < 0.20 ≤ MEDIUM < 0.45 ≤ HIGH < 0.75 ≤ CRITICAL (CRITICAL is blocked).
  Edit `RISK_LEVELS` in `config.py`.
- **OFHXOR:** the (K\*, D1, D2) block structure is an interpretation of the architecture
  diagram; adjust `ofhxor.py` if your specification differs.
- **Security:** XOR-based encryption is for research and benchmarking. For real patient
  data use an authenticated cipher such as AES-GCM.
- Detection is binary (normal vs. attack); one HMOFS + PSL detector is trained per dataset
  because the schemas differ.
