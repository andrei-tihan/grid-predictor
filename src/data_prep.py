# src/data_prep.py

# Script to load, clean, and merge Alberta generation and load CSVs.
# Outputs a cleaned CSV with consistent hourly timestamps and engineered features.

from pathlib import Path
import pandas as pd
import numpy as np

TZ = "America/Edmonton"

def _normalize_col(c):
    if not isinstance(c, str):
        return c
    s = c.strip().lower()
    s = s.replace("(", "").replace(")", "")
    s = s.replace("/", "_").replace("-", "_").replace(".", "").replace(",", "")
    s = "_".join(s.split())
    return s

def _find_by_keywords(cols, keywords_list):
    """
    Return first column name that contains all keywords in one of the keyword lists.
    keywords_list: list of keyword-lists (priority)
    """
    norm = {c: _normalize_col(c) for c in cols}
    for keywords in keywords_list:
        for c, nc in norm.items():
            if all(k in nc for k in keywords):
                return c
    return None

def _to_numeric(s):
    return pd.to_numeric(s.astype(str).str.replace(",","").str.strip(), errors="coerce")

def load_generation_csv(path: Path):
    df = pd.read_csv(path)
    cols = list(df.columns)

    # find date col
    date_col = _find_by_keywords(cols, [["date"], ["time"], ["date_time"], ["date_-_mst"], ["date_mst"]]) or cols[0]

    # find main columns
    total_gen_col = _find_by_keywords(cols, [["total","generation"], ["total_generation"], ["total_gen"], ["total"]])
    wind_col = _find_by_keywords(cols, [["wind"]])
    solar_col = _find_by_keywords(cols, [["solar"]])
    other_col = _find_by_keywords(cols, [["other"]])

    # Build df with only relevant columns if present
    keep = [date_col]
    for c in (total_gen_col, wind_col, solar_col, other_col):
        if c and c not in keep:
            keep.append(c)
    df2 = df[keep].copy()

    # parse timestamp
    df2['timestamp'] = pd.to_datetime(df2[date_col], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
    # localize to TZ (strings appear to be local MST)
    if df2['timestamp'].dt.tz is None:
        try:
            df2['timestamp'] = df2['timestamp'].dt.tz_localize(TZ, ambiguous='infer', nonexistent='shift_forward')
        except Exception:
            df2['timestamp'] = df2['timestamp'].dt.tz_localize(TZ, ambiguous='NaT', nonexistent='NaT')
    else:
        df2['timestamp'] = df2['timestamp'].dt.tz_convert(TZ)

    # numeric fields
    df2['total_generation_mw'] = _to_numeric(df2[total_gen_col]) if total_gen_col else np.nan
    df2['wind_mw'] = _to_numeric(df2[wind_col]) if wind_col else np.nan
    df2['solar_mw'] = _to_numeric(df2[solar_col]) if solar_col else np.nan
    if other_col:
        df2['other_orig_mw'] = _to_numeric(df2[other_col])
    else:
        df2['other_orig_mw'] = np.nan

    out = df2[['timestamp','total_generation_mw','wind_mw','solar_mw','other_orig_mw']].copy()
    return out

def load_load_csv(path: Path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    date_col = _find_by_keywords(cols, [["date"], ["time"], ["date_time"], ["date_-_mst"], ["date_mst"]]) or cols[0]
    load_col = _find_by_keywords(cols, [["ail"], ["load"], ["demand"]])
    if load_col is None:
        raise ValueError(f"Could not find AIL/load column in {path}. Columns: {cols}")
    df2 = df[[date_col, load_col]].copy()
    df2['timestamp'] = pd.to_datetime(df2[date_col], errors='coerce')
    if df2['timestamp'].dt.tz is None:
        try:
            df2['timestamp'] = df2['timestamp'].dt.tz_localize(TZ, ambiguous='infer', nonexistent='shift_forward')
        except Exception:
            df2['timestamp'] = df2['timestamp'].dt.tz_localize(TZ, ambiguous='NaT', nonexistent='NaT')
    else:
        df2['timestamp'] = df2['timestamp'].dt.tz_convert(TZ)
    df2['load_mw'] = _to_numeric(df2[load_col])
    return df2[['timestamp','load_mw']]

def merge_and_clean(gen_df, load_df, out_path: Path):
    # Outer merge to preserve any timestamps that might be missing in one
    df = pd.merge(gen_df, load_df, on='timestamp', how='outer', sort=True)

    # Ensure datetime sorted, dedupe
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

    # Compute other_gen (if wind/solar present)
    if 'wind_mw' in df.columns and 'solar_mw' in df.columns:
        df['wind_mw'] = df['wind_mw'].fillna(0.0)
        df['solar_mw'] = df['solar_mw'].fillna(0.0)
        df['other_gen_mw'] = df['total_generation_mw'] - (df['wind_mw'] + df['solar_mw'])
    else:
        df['other_gen_mw'] = df.get('other_orig_mw', np.nan)

    # Compute balance
    df['balance_mw'] = df['total_generation_mw'] - df['load_mw']

    # Check continuity and fill small gaps
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    expected_idx = pd.date_range(ts_min, ts_max, freq='h', tz=TZ)
    if len(expected_idx) != df.shape[0]:
        # reindex & interpolate small gaps
        df = df.set_index('timestamp').reindex(expected_idx)
        df.index.name = 'timestamp'
        numeric_cols = ['total_generation_mw','load_mw','wind_mw','solar_mw','other_gen_mw','balance_mw']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = df[c].interpolate(limit=2)
        df = df.reset_index()

    # Print basic sanity
    missing_gen = df['total_generation_mw'].isna().sum()
    missing_load = df['load_mw'].isna().sum()
    print("Sanity checks after merge:")
    print(f"  rows: {len(df)}")
    print(f"  missing generation rows: {missing_gen}")
    print(f"  missing load rows: {missing_load}")

    # Feature engineering (basic)
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    for lag in [1,24,168]:
        df[f'balance_lag{lag}'] = df['balance_mw'].shift(lag)
    for w in [24,168]:
        df[f'balance_roll_{w}'] = df['balance_mw'].rolling(w).mean().shift(1)

    # Save a clean CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned CSV to {out_path}")
    return df

def load_and_prepare(gen_csv, load_csv, out_path=Path("data/processed/alberta_hourly_clean.csv")):
    gen = Path(gen_csv)
    load = Path(load_csv)
    if not gen.exists() or not load.exists():
        raise FileNotFoundError("Generation / Load CSV not found at paths provided.")
    gen_df = load_generation_csv(gen)
    load_df = load_load_csv(load)
    return merge_and_clean(gen_df, load_df, out_path)