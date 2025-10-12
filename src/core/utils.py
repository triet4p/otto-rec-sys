import cudf
import pandas as pd
import polars as pl

from .config import TYPE_LABELS_MAPPING

def parquet_to_pd_df(parquet_file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file_path)
    
    # Reduce ts memory
    df['ts'] = (df['ts'] / 1000).astype('int32')
    df['type'] = df['type'].map(TYPE_LABELS_MAPPING).astype('int8')
    
    return df

def pd_df_to_cudf(df: pd.DataFrame) -> cudf.DataFrame:
    return cudf.DataFrame(df)

def load_raw_data_parquet(file_pattern):
    return pl.scan_parquet(file_pattern) \
        .with_columns(pl.col('type').replace(TYPE_LABELS_MAPPING).cast(pl.Int8)) \
        .with_columns((pl.col('ts') / 1000).cast(pl.Int32)) \
        .collect()