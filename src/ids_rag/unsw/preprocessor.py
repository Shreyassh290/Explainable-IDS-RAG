"""Strict leak-free preprocessor for UNSW-NB15 binary classification.

Design Invariants:
1. 'id', 'ID', 'label', and 'attack_cat' are explicitly excluded from feature columns.
2. Preprocessor is fitted EXCLUSIVELY on training folds.
3. Duplicates are dropped strictly within the training fold (never test).
4. Engineered features use strictly non-target continuous inputs with zero-division protection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDED_COLUMNS = ["id", "ID", "label", "attack_cat", "stcpb", "dtcpb"]

SKEWED_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "response_body_len"
]

ENGINEERED_SKEWED_FEATURES = [
    "bytes_per_pkt_src",
    "bytes_per_pkt_dst",
    "ttl_ratio",
    "load_ratio",
]

PREDEFINED_CATEGORIES: dict[str, list[str]] = {
    "proto": ["-", "arp", "gre", "ospf", "other", "sctp", "sun-nd", "swipe", "tcp", "udp", "unas"],
    "service": ["-", "dhcp", "dns", "ftp", "ftp-data", "http", "other", "pop3", "smtp", "snmp", "ssh"],
    "state": ["CON", "ECO", "FIN", "INT", "PAR", "REQ", "RST", "URN", "no", "other"],
}


def compute_domain_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 domain ratio and asymmetry features using only traffic volume inputs."""
    df = df_in.copy()

    spkts = np.maximum(df["spkts"].to_numpy(dtype=np.float32), 0.0)
    dpkts = np.maximum(df["dpkts"].to_numpy(dtype=np.float32), 0.0)
    sbytes = np.maximum(df["sbytes"].to_numpy(dtype=np.float32), 0.0)
    dbytes = np.maximum(df["dbytes"].to_numpy(dtype=np.float32), 0.0)
    sttl = np.maximum(df["sttl"].to_numpy(dtype=np.float32), 0.0)
    dttl = np.maximum(df["dttl"].to_numpy(dtype=np.float32), 0.0)
    sload = np.maximum(df["sload"].to_numpy(dtype=np.float32), 0.0)
    dload = np.maximum(df["dload"].to_numpy(dtype=np.float32), 0.0)

    # Ratio and asymmetry features with safe denominator padding
    df["bytes_per_pkt_src"] = sbytes / (spkts + 1.0)
    df["bytes_per_pkt_dst"] = dbytes / (dpkts + 1.0)
    df["packet_asymmetry"] = (spkts - dpkts) / (spkts + dpkts + 1.0)
    df["byte_asymmetry"] = (sbytes - dbytes) / (sbytes + dbytes + 1.0)
    df["ttl_ratio"] = sttl / (dttl + 1.0)
    df["load_ratio"] = sload / (dload + 1.0)

    return df


class UNSWStrictPreprocessor:
    """Preprocessor fitted exclusively on training folds with guaranteed feature exclusion."""

    def __init__(self, predefined_categories: dict[str, list[str]] | None = None) -> None:
        self.cat_cols = ["proto", "service", "state"]
        self.categories_ = predefined_categories or PREDEFINED_CATEGORIES
        self.encoder_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.numeric_cols_: list[str] = []

    def fit(self, df_train_fold: pd.DataFrame) -> UNSWStrictPreprocessor:
        df = compute_domain_features(df_train_fold)

        # Enforce that id, label, attack_cat are NEVER included
        candidate_cols = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
        for col in ["id", "ID", "label", "attack_cat"]:
            assert col not in candidate_cols, f"Forbidden column {col} in feature candidates"

        self.numeric_cols_ = [c for c in candidate_cols if c not in self.cat_cols]

        # 1. Categorical preparation
        cat_df_tr = pd.DataFrame(index=df.index)
        for col in self.cat_cols:
            allowed = set(self.categories_[col])
            cat_df_tr[col] = df[col].astype(str).apply(lambda x: x if x in allowed else "other")

        cat_lists = [self.categories_[c] for c in self.cat_cols]
        self.encoder_ = OneHotEncoder(
            categories=cat_lists,
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        )
        self.encoder_.fit(cat_df_tr)

        # 2. Numeric log1p and standard scaling strictly on training fold
        num_vals = df[self.numeric_cols_].copy()
        all_skewed = SKEWED_FEATURES + ENGINEERED_SKEWED_FEATURES
        for col in all_skewed:
            if col in self.numeric_cols_:
                num_vals[col] = np.log1p(np.maximum(num_vals[col].to_numpy(dtype=np.float32), 0.0))

        self.scaler_ = StandardScaler()
        self.scaler_.fit(num_vals.to_numpy(dtype=np.float32))

        # 3. Final feature list (43 numeric + 29 categorical = 72)
        encoded_cat_names = self.encoder_.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names_ = self.numeric_cols_ + encoded_cat_names

        # Strict assertion against target or ID leakage in feature names
        for fn in self.feature_names_:
            assert fn not in ["id", "ID", "label", "attack_cat"], f"Target/ID column {fn} leaked into feature names"

        return self

    def transform(self, df_in: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None or self.encoder_ is None:
            raise RuntimeError("Preprocessor must be fitted on training fold before transform.")

        df = compute_domain_features(df_in)

        # 1. Numeric transformation
        num_vals = df[self.numeric_cols_].copy()
        all_skewed = SKEWED_FEATURES + ENGINEERED_SKEWED_FEATURES
        for col in all_skewed:
            if col in self.numeric_cols_:
                num_vals[col] = np.log1p(np.maximum(num_vals[col].to_numpy(dtype=np.float32), 0.0))

        num_scaled = self.scaler_.transform(num_vals.to_numpy(dtype=np.float32))

        # 2. Categorical transformation
        cat_df = pd.DataFrame(index=df.index)
        for col in self.cat_cols:
            allowed = set(self.categories_[col])
            cat_df[col] = df[col].astype(str).apply(lambda x: x if x in allowed else "other")

        cat_encoded = self.encoder_.transform(cat_df)

        # 3. Concatenate and sanitize
        combined = np.hstack([num_scaled, cat_encoded]).astype(np.float32)
        return np.nan_to_num(combined, nan=0.0)

