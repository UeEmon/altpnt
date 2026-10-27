"""VOACAP（Method 30）出力の全項目パーサ.

CHUCK の parse_output は REL/SNR など一部だけを取り出す。上空波の測位では
伝搬モード・遅延・仮想反射高度・マルチパス確率も要るので、生出力から全行を読む。
各時刻ブロックは「時刻 MUF f1 f2 ... FREQ」の行で始まり、以降の行は末尾のラベルで識別する。
数値は 1 桁目から 5 文字幅の固定長。
"""
from __future__ import annotations

LABELS = {
    "MODE": ("mode", str), "TANGLE": ("takeoff_deg", float), "DELAY": ("delay_ms", float),
    "V HITE": ("vhite_km", float), "MUFday": ("mufday", float), "LOSS": ("loss_db", float),
    "DBU": ("dbu", float), "S DBW": ("s_dbw", float), "N DBW": ("n_dbw", float),
    "SNR": ("snr_dbhz", float), "RPWRG": ("rpwrg_db", float), "REL": ("rel", float),
    "MPROB": ("mprob", float), "S PRB": ("sprob", float),
}


def _field(line: str, i: int) -> str:
    return line[1 + i * 5: 1 + (i + 1) * 5]


def parse_voacap(raw: str) -> list[dict]:
    """[{hour_utc, muf_mhz, rows: [{freq_mhz, mode, delay_ms, vhite_km, rel, snr_dbhz, ...}]}] を時刻順に返す.

    rows[0] は VOACAP が MUF 列に出す値（MUF での伝搬）なので除外し、指定周波数だけを返す。
    """
    blocks: list[dict] = []
    cur = None
    for line in raw.splitlines():
        s = line.rstrip()
        if s.endswith(" FREQ"):
            try:
                vals = [float(_field(line, i)) for i in range(0, 13)]
            except ValueError:
                cur = None
                continue
            freqs = [f for f in vals[2:] if f > 0]
            cur = {"hour_utc": int(vals[0]) % 24, "muf_mhz": vals[1], "rows": [{"freq_mhz": f} for f in freqs]}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        label = next((k for k in LABELS if s.endswith(" " + k)), None)
        if label is None:
            continue
        key, typ = LABELS[label]
        for j, row in enumerate(cur["rows"]):
            tok = _field(line, j + 2).strip()     # 0: 時刻列, 1: MUF 列, 2 以降: 指定周波数
            if typ is str:
                row[key] = tok or None
            else:
                try:
                    row[key] = float(tok)
                except ValueError:
                    row[key] = None
    if not blocks:
        raise ValueError("VOACAP 出力に時刻ブロックが見つかりません")
    return sorted(blocks, key=lambda b: b["hour_utc"])
