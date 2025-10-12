import polars as pl

def load_covisit_matrix(path_pattern, 
                        covisit_matrix_type: str,
                        top_k=100):
    """
    Tải tất cả các phần của một ma trận co-visitation, nối chúng lại,
    và tính toán thứ hạng cho mỗi gợi ý.
    """
    df = pl.scan_parquet(path_pattern) \
        .sort(['aid_x', 'wgt'], descending=[False, True]) \
        .with_columns(pl.arange(0, pl.len()).over('aid_x').alias('rank').cast(pl.UInt16)) \
        .filter(pl.col('rank') < top_k) \
        .collect()
    
    # --- THÊM BƯỚC DỌN DẸP ---
    # Kiểm tra xem cột có tồn tại không rồi mới xóa
    if '__index_level_0__' in df.columns:
        df = df.drop('__index_level_0__')

    df = df.rename({'aid_x': 'aid', 'aid_y': 'candidate_aid', 
                    'wgt': f'wgt_{covisit_matrix_type}', 
                    'rank': f'rank_{covisit_matrix_type}'})
        
    return df