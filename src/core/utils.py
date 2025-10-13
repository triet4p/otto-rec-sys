import polars as pl

from .config import TYPE_LABELS_MAPPING

def load_raw_data_parquet(file_pattern):
    return pl.scan_parquet(file_pattern) \
        .with_columns(pl.col('type').replace(TYPE_LABELS_MAPPING).cast(pl.Int8)) \
        .with_columns((pl.col('ts') / 1000).cast(pl.Int32)) \
        .collect()