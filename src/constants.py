# Model dimensions for Mamba-130M
D_MODEL = 768
D_STATE = 16
D_CONV = 4
EXPAND = 2
D_INNER = EXPAND * D_MODEL # 1536
DT_RANK = "auto" # or 48 for 130M
