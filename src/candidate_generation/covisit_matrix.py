import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import cudf
import gc
import glob
from src.core.utils import pd_df_to_cudf, parquet_to_pd_df

_DF_CACHE: Dict[str, pd.DataFrame] = {}

def cache_files(files: List[str]): 
    global _DF_CACHE
    for file in files:
        _DF_CACHE[file] = parquet_to_pd_df(file)
        
def clear_cache():
    global _DF_CACHE
    _DF_CACHE.clear()
    
CO_VISIT_MATRIX_OUT_PATH_TEMPLATE = '{output_dir}/top_{top_k}_{matrix}_{part}.pqt'
    
def calculate_cart_order_matrix(
    files: list[str],
    read_ct: int,
    num_chunk: int,
    fix_size: int,
    disk_pieces: int,
    last_hold_aids_each_session: int,
    valid_time: int,
    top_k: int,
    type_weight: dict,
    save_parquet: bool=False,
    output_dir: str|None=None
) -> dict:
    piece_size = fix_size / disk_pieces
    chunk_size = int(np.ceil(len(files) / num_chunk))
    saved_files = []
    for part in range(disk_pieces):
        # Mỗi vòng to sẽ xử lý 1/4 dữ liệu
        print('='*20)
        print(f'### DISK PART {part + 1}')
    
        for chunk_idx in range(num_chunk):
            # Qua từng chunk nhỏ để giảm RAM
            start_file_idx = chunk_idx * chunk_size
            end_file_idx = min((chunk_idx + 1) * chunk_size, len(files))
            print(f'Processing from file {start_file_idx} to {end_file_idx} in group of {read_ct}')
    
            for k in range(start_file_idx, end_file_idx, read_ct):
                # Inner chunk
                # Đọc 5 file một lúc từ cache, đẩy lên GPU và nối lại
                df = [pd_df_to_cudf(_DF_CACHE[files[k]])]
                for i in range(1, read_ct):
                    if k + i < end_file_idx: 
                        df.append(pd_df_to_cudf(_DF_CACHE[files[k+i]]))
                df = cudf.concat(df, ignore_index=True, axis=0)
    
                # Sắp xếp theo session và thời gian (giảm dần)
                df = df.sort_values(['session','ts'], ascending=[True,False])
                
                # --- Tối ưu hóa quan trọng ---
                # Chỉ giữ lại 30 hành động cuối cùng của mỗi session
                df = df.reset_index(drop=True)
                df['n'] = df.groupby('session').cumcount()
                df = df.loc[df.n<last_hold_aids_each_session].drop('n',axis=1)
    
                # Tự merge một dataframe với chính nó dựa trên session
                # Kết quả: tạo ra tất cả các cặp (aid_x, aid_y) có thể có trong cùng 1 session
                df = df.merge(df, on='session')
    
                # --- Lọc các cặp không hợp lệ ---
                # 1. Chỉ giữ các cặp có thời gian cách nhau không quá 1 ngày (24*60*60 giây)
                # 2. Loại bỏ các cặp mà aid_x giống hệt aid_y
                df = df.loc[((df.ts_x - df.ts_y).abs()< valid_time) & (df.aid_x != df.aid_y)]
    
                # Chỉ tính cho các aid_x thuộc phần hiện tại
                df = df.loc[(df.aid_x >= part*piece_size) & (df.aid_x < (part+1)*piece_size)]
    
                # --- Gán trọng số và tính toán ---
                # Chỉ giữ các cột cần thiết và loại bỏ các cặp trùng lặp
                df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
                # Dùng .map() để gán trọng số dựa trên 'type' của sản phẩm thứ hai (aid_y)
                df['wgt'] = df.type_y.map(type_weight)
                df = df[['aid_x','aid_y','wgt']]
                df.wgt = df.wgt.astype('float32')
    
                # Nhóm theo cặp (aid_x, aid_y) và cộng tất cả các trọng số lại
                df = df.groupby(['aid_x','aid_y']).wgt.sum()
    
                # Combine inner chunk
                if k == start_file_idx:
                    inner_tmp = df
                else:
                    inner_tmp = inner_tmp.add(df, fill_value=0)
    
                print(k,', ',end='')
            print("="*10)
            # Combine outer chunk
            if start_file_idx == 0:
                outer_tmp = inner_tmp
            else:
                outer_tmp = outer_tmp.add(inner_tmp, fill_value=0)
            del inner_tmp, df
            gc.collect()
    
        # Convert matrix to dict
        outer_tmp = outer_tmp.reset_index()
        outer_tmp = outer_tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    
        outer_tmp = outer_tmp.reset_index(drop=True)
        outer_tmp['n'] = outer_tmp.groupby('aid_x')['aid_y'].cumcount()
        outer_tmp = outer_tmp.loc[outer_tmp.n < top_k].drop('n', axis=1)
    
        # Save to disk
        if save_parquet:
            out_path = CO_VISIT_MATRIX_OUT_PATH_TEMPLATE.format(
                output_dir=output_dir if output_dir else '/kaggle/working',
                top_k=top_k,
                matrix='cart_order',
                part=part
            )
            outer_tmp.to_pandas().to_parquet(out_path)
            saved_files.append(out_path)

    return {
        'num_part': len(saved_files),
        'matrix_type': 'cart_order',
        'saved_files': saved_files,
        'top_k': top_k
    }
    
def calculate_buy2buy_matrix(
    files: list[str],
    read_ct: int,
    num_chunk: int,
    fix_size: int,
    disk_pieces: int,
    last_hold_aids_each_session: int,
    valid_time: int,
    top_k: int,
    save_parquet: bool=False,
    output_dir: str|None=None
):
    piece_size = fix_size / disk_pieces
    chunk_size = int(np.ceil(len(files) / num_chunk))
    saved_files = []
    for part in range(disk_pieces):
        # Mỗi vòng to sẽ xử lý 1/4 dữ liệu
        print('='*20)
        print(f'### DISK PART {part + 1}')
    
        for chunk_idx in range(num_chunk):
            # Qua từng chunk nhỏ để giảm RAM
            start_file_idx = chunk_idx * chunk_size
            end_file_idx = min((chunk_idx + 1) * chunk_size, len(files))
            print(f'Processing from file {start_file_idx} to {end_file_idx} in group of {read_ct}')
    
            for k in range(start_file_idx, end_file_idx, read_ct):
                # Inner chunk
                # Đọc 5 file một lúc từ cache, đẩy lên GPU và nối lại
                df = [pd_df_to_cudf(_DF_CACHE[files[k]])]
                for i in range(1, read_ct):
                    if k + i < end_file_idx: 
                        df.append(pd_df_to_cudf(_DF_CACHE[files[k+i]]))
                df = cudf.concat(df, ignore_index=True, axis=0)
                df = df.loc[df['type'].isin([1,2])] # ONLY WANT CARTS AND ORDERS
                # Sắp xếp theo session và thời gian (giảm dần)
                df = df.sort_values(['session','ts'], ascending=[True,False])
                
                # --- Tối ưu hóa quan trọng ---
                # Chỉ giữ lại 30 hành động cuối cùng của mỗi session
                df = df.reset_index(drop=True)
                df['n'] = df.groupby('session').cumcount()
                df = df.loc[df.n<last_hold_aids_each_session].drop('n',axis=1)
    
                # Tự merge một dataframe với chính nó dựa trên session
                # Kết quả: tạo ra tất cả các cặp (aid_x, aid_y) có thể có trong cùng 1 session
                df = df.merge(df, on='session')
    
                # --- Lọc các cặp không hợp lệ ---
                # 1. Chỉ giữ các cặp có thời gian cách nhau không quá 1 ngày (24*60*60 giây)
                # 2. Loại bỏ các cặp mà aid_x giống hệt aid_y
                df = df.loc[((df.ts_x - df.ts_y).abs()< valid_time) & (df.aid_x != df.aid_y)]
    
                # Chỉ tính cho các aid_x thuộc phần hiện tại
                df = df.loc[(df.aid_x >= part*piece_size) & (df.aid_x < (part+1)*piece_size)]
    
                # --- Gán trọng số và tính toán ---
                # Chỉ giữ các cột cần thiết và loại bỏ các cặp trùng lặp
                df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
                # Dùng .map() để gán trọng số dựa trên 'type' của sản phẩm thứ hai (aid_y)
                df['wgt'] = 1
                df = df[['aid_x','aid_y','wgt']]
                df.wgt = df.wgt.astype('float32')
    
                # Nhóm theo cặp (aid_x, aid_y) và cộng tất cả các trọng số lại
                df = df.groupby(['aid_x','aid_y']).wgt.sum()
    
                # Combine inner chunk
                if k == start_file_idx:
                    inner_tmp = df
                else:
                    inner_tmp = inner_tmp.add(df, fill_value=0)
    
                print(k,', ',end='')
            print("="*10)
            # Combine outer chunk
            if start_file_idx == 0:
                outer_tmp = inner_tmp
            else:
                outer_tmp = outer_tmp.add(inner_tmp, fill_value=0)
            del inner_tmp, df
            gc.collect()
    
        # Convert matrix to dict
        outer_tmp = outer_tmp.reset_index()
        outer_tmp = outer_tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    
        outer_tmp = outer_tmp.reset_index(drop=True)
        outer_tmp['n'] = outer_tmp.groupby('aid_x')['aid_y'].cumcount()
        outer_tmp = outer_tmp.loc[outer_tmp.n < top_k].drop('n', axis=1)
    
        # Save to disk
        if save_parquet:
            out_path = CO_VISIT_MATRIX_OUT_PATH_TEMPLATE.format(
                output_dir=output_dir if output_dir else '/kaggle/working',
                top_k=top_k,
                matrix='buy2buy',
                part=part
            )
            outer_tmp.to_pandas().to_parquet(out_path)
            saved_files.append(out_path)

    return {
        'num_part': len(saved_files),
        'matrix_type': 'buy2buy',
        'saved_files': saved_files,
        'top_k': top_k
    }
    
def calculate_clicks_matrix(
    files: list[str],
    read_ct: int,
    num_chunk: int,
    fix_size: int,
    disk_pieces: int,
    last_hold_aids_each_session: int,
    valid_time: int,
    top_k: int,
    time_weighted_func: callable,
    save_parquet: bool=False,
    output_dir: str|None=None
):
    piece_size = fix_size / disk_pieces
    chunk_size = int(np.ceil(len(files) / num_chunk))
    saved_files = []
    for part in range(disk_pieces):
        # Mỗi vòng to sẽ xử lý 1/4 dữ liệu
        print('='*20)
        print(f'### DISK PART {part + 1}')
    
        for chunk_idx in range(num_chunk):
            # Qua từng chunk nhỏ để giảm RAM
            start_file_idx = chunk_idx * chunk_size
            end_file_idx = min((chunk_idx + 1) * chunk_size, len(files))
            print(f'Processing from file {start_file_idx} to {end_file_idx} in group of {read_ct}')
    
            for k in range(start_file_idx, end_file_idx, read_ct):
                # Inner chunk
                # Đọc 5 file một lúc từ cache, đẩy lên GPU và nối lại
                df = [pd_df_to_cudf(_DF_CACHE[files[k]])]
                for i in range(1, read_ct):
                    if k + i < end_file_idx: 
                        df.append(pd_df_to_cudf(_DF_CACHE[files[k+i]]))
                df = cudf.concat(df, ignore_index=True, axis=0)
    
                # Sắp xếp theo session và thời gian (giảm dần)
                df = df.sort_values(['session','ts'], ascending=[True,False])
                
                # --- Tối ưu hóa quan trọng ---
                # Chỉ giữ lại 30 hành động cuối cùng của mỗi session
                df = df.reset_index(drop=True)
                df['n'] = df.groupby('session').cumcount()
                df = df.loc[df.n<last_hold_aids_each_session].drop('n',axis=1)
    
                # Tự merge một dataframe với chính nó dựa trên session
                # Kết quả: tạo ra tất cả các cặp (aid_x, aid_y) có thể có trong cùng 1 session
                df = df.merge(df, on='session')
    
                # --- Lọc các cặp không hợp lệ ---
                # 1. Chỉ giữ các cặp có thời gian cách nhau không quá 1 ngày (24*60*60 giây)
                # 2. Loại bỏ các cặp mà aid_x giống hệt aid_y
                df = df.loc[((df.ts_x - df.ts_y).abs()< valid_time) & (df.aid_x != df.aid_y)]
    
                # Chỉ tính cho các aid_x thuộc phần hiện tại
                df = df.loc[(df.aid_x >= part*piece_size) & (df.aid_x < (part+1)*piece_size)]
    
                # --- Gán trọng số và tính toán ---
                # Chỉ giữ các cột cần thiết và loại bỏ các cặp trùng lặp
                df = df[['session', 'aid_x', 'aid_y','type_y', 'ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
                df['wgt'] = time_weighted_func(df['ts_x'])
                df = df[['aid_x','aid_y','wgt']]
                df.wgt = df.wgt.astype('float32')
    
                # Nhóm theo cặp (aid_x, aid_y) và cộng tất cả các trọng số lại
                df = df.groupby(['aid_x','aid_y']).wgt.sum()
    
                # Combine inner chunk
                if k == start_file_idx:
                    inner_tmp = df
                else:
                    inner_tmp = inner_tmp.add(df, fill_value=0)
    
                print(k,', ',end='')
            print("="*10)
            # Combine outer chunk
            if start_file_idx == 0:
                outer_tmp = inner_tmp
            else:
                outer_tmp = outer_tmp.add(inner_tmp, fill_value=0)
            del inner_tmp, df
            gc.collect()
    
        # Convert matrix to dict
        outer_tmp = outer_tmp.reset_index()
        outer_tmp = outer_tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    
        outer_tmp = outer_tmp.reset_index(drop=True)
        outer_tmp['n'] = outer_tmp.groupby('aid_x')['aid_y'].cumcount()
        outer_tmp = outer_tmp.loc[outer_tmp.n < top_k].drop('n', axis=1)
    
        # Save to disk
        if save_parquet:
            out_path = CO_VISIT_MATRIX_OUT_PATH_TEMPLATE.format(
                output_dir=output_dir if output_dir else '/kaggle/working',
                top_k=top_k,
                matrix='clicks',
                part=part
            )
            outer_tmp.to_pandas().to_parquet(out_path)
            saved_files.append(out_path)

    return {
        'num_part': len(saved_files),
        'matrix_type': 'clicks',
        'saved_files': saved_files,
        'top_k': top_k
    }
    
def generate_candidate(file_pattern: str,
                       output_dir: str,
                       *,
                       carts_orders_params: Dict[str, Any],
                       buy2buy_params: Dict[str, Any],
                       clicks_params: Dict[str, Any]):
    files = glob.glob(file_pattern)
    cache_files(files)
    
    os.makedirs(output_dir, exist_ok=True)
    
    calculate_cart_order_matrix(**carts_orders_params)
    calculate_buy2buy_matrix(**buy2buy_params)
    calculate_clicks_matrix(**clicks_params)
    
    clear_cache()