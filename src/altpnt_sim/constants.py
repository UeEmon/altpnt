"""物理定数と既定パラメータ."""

C = 299_792_458.0            # 光速 [m/s]
K_BOLTZ = 1.380649e-23       # ボルツマン定数 [J/K]
T0 = 290.0                   # 基準雑音温度 [K]
R_EARTH = 6_371_000.0        # 地球半径 [m]
K_FACTOR = 4.0 / 3.0         # 標準大気の等価地球半径係数

# Gサイズソノブイ筐体（引継書の値。提案書の「長さ約91cm」とは不一致 → docs/OPEN_ISSUES.md）
G_SIZE_DIAMETER_M = 0.124
G_SIZE_LENGTH_M = 0.368

VHF_LINK_RATE_BPS = 120_000  # VHFアップリンク帯域上限
