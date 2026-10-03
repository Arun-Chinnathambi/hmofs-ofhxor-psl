"""Central configuration constants (values taken from the system architecture)."""

THETA = 0.45  # PSL decision threshold: attack iff fused posterior >= THETA

# (upper probability bound, risk level, action) - checked in order
RISK_LEVELS = [
    (0.20, "LOW", "ALLOW"),
    (THETA, "MEDIUM", "ALLOW_MONITOR"),
    (0.75, "HIGH", "QUARANTINE"),
    (1.01, "CRITICAL", "BLOCK"),
]

# Dataset column conventions (WUSTL-EHMS-2020 / ECU-IoHT)
LABEL_CANDIDATES = ["Label", "label", "Attack Category", "Type", "type", "Class"]
BENIGN_WORDS = {"normal", "benign", "0", "none", "no attack"}

VITALS = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate"]
NET = ["SrcBytes", "DstBytes", "SrcLoad", "DstLoad", "SrcGap", "DstGap", "SIntPkt",
       "DIntPkt", "SrcJitter", "DstJitter", "sMaxPktSz", "dMaxPktSz", "sMinPktSz",
       "dMinPktSz", "Dur", "Trans", "TotPkts", "TotBytes", "Load", "Loss", "Rate",
       "Packet_num"]

KEY_LENGTH = 32          # bytes in the optimal XOR key K*
FEEDBACK_RETRAIN_EVERY = 60
FEEDBACK_UNCERTAINTY = 0.10
