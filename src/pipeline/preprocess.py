import gc
import os
import faiss
import numpy as np
import polars as pl
import tqdm
import lightgbm as lgb
from src.candidate_generation.gnn import generate_candidates_from_gnn
import src.core.config as cfg


def process_chunk(history_chunk: pl.DataFrame, 
                  popular_items_df: pl.DataFrame, 
                  df_clicks: pl.DataFrame, df_buys: pl.DataFrame, df_buy2buy: pl.DataFrame,
                  use_gnn_embedding: bool = False,
                  embedding_df: pl.DataFrame|None = None, 
                  faiss_index: faiss.Index|None = None, 
                  idx2aid_faiss: dict|None = None) -> pl.DataFrame:
    """
    Xử lý một khối (chunk) session: tạo ứng viên từ lịch sử và co-visitation,
    sau đó tạo các đặc trưng về nguồn gốc (rank, wgt).

    Args:
        history_chunk (pl.DataFrame): DataFrame chứa ['session', 'aid'] cho một phần các session.
        df_clicks (pl.DataFrame): Ma trận co-visitation clicks.
        df_buys (pl.DataFrame): Ma trận co-visitation buys.
        df_buy2buy (pl.DataFrame): Ma trận co-visitation buy2buy.

    Returns:
        pl.DataFrame: DataFrame chứa các ứng viên và đặc trưng cho chunk đó.
    """
    # 1. Tạo ứng viên từ Co-visitation bằng cách join với lịch sử của chunk
    candidates_clicks_chunk = history_chunk.join(df_clicks, on='aid', how='inner')
    candidates_buys_chunk = history_chunk.join(df_buys, on='aid', how='inner')
    candidates_buy2buy_chunk = history_chunk.join(df_buy2buy, on='aid', how='inner')

    # 2. Tạo ứng viên từ lịch sử
    candidates_history_chunk = history_chunk.rename({'aid': 'candidate_aid'})
    # 3. Tạo ứng viên phổ biến CHỈ CHO CHUNK NÀY
    sessions_in_chunk = history_chunk.select('session').unique()
    candidates_popular_chunk = sessions_in_chunk.join(popular_items_df, how='cross')
    
<<<<<<< Updated upstream
    candidates_gnn_chunk = generate_candidates_from_gnn(history_chunk, embedding_df, faiss_index, idx2aid_faiss)
    #candidates_gnn_chunk = candidates_gnn_chunk.with_columns(pl.col('session').cast(pl.Int32))
    # 4. Tổng hợp tất cả ứng viên cho chunk này
=======
>>>>>>> Stashed changes
    candidates_df_chunk = pl.concat([
        candidates_history_chunk.select(['session', 'candidate_aid']),
        candidates_popular_chunk, # Thêm nguồn popular ở đây 
        candidates_clicks_chunk.select(['session', 'candidate_aid']),
        candidates_buys_chunk.select(['session', 'candidate_aid']),
        candidates_buy2buy_chunk.select(['session', 'candidate_aid']),
    ]).unique(subset=['session', 'candidate_aid'], keep='first')
    
    if use_gnn_embedding:
        candidates_gnn_chunk = generate_candidates_from_gnn(history_chunk, embedding_df, faiss_index, idx2aid_faiss)
        candidates_gnn_chunk = candidates_gnn_chunk.with_columns(pl.col('session').cast(pl.Int32))
        # 4. Tổng hợp tất cả ứng viên cho chunk này
        candidates_df_chunk = pl.concat([
            candidates_df_chunk,
            candidates_gnn_chunk.select(['session', 'candidate_aid'])
        ]).unique(subset=['session', 'candidate_aid'], keep='first')
    
    # 4. Tạo đặc trưng từ nguồn gốc (aggregate và join ngược lại)
    # Đặc trưng từ lịch sử
    candidates_history_chunk = candidates_history_chunk.with_columns(pl.lit(1).cast(pl.UInt8).alias('source_history'))
    candidates_df_chunk = candidates_df_chunk.join(candidates_history_chunk, on=['session', 'candidate_aid'], how='left')
    
    # Đặc trưng từ co-visitation (lấy rank tốt nhất và score cao nhất)
    agg_clicks_chunk = candidates_clicks_chunk.group_by('session', 'candidate_aid', maintain_order=False).agg(
        pl.col('rank_clicks').min(), pl.col('wgt_clicks').max()
    )
    agg_buys_chunk = candidates_buys_chunk.group_by('session', 'candidate_aid', maintain_order=False).agg(
        pl.col('rank_buys').min(), pl.col('wgt_buys').max()
    )
    agg_buy2buy_chunk = candidates_buy2buy_chunk.group_by('session', 'candidate_aid', maintain_order=False).agg(
        pl.col('rank_buy2buy').min(), pl.col('wgt_buy2buy').max()
    )
    
    candidates_df_chunk = candidates_df_chunk.join(agg_clicks_chunk, on=['session', 'candidate_aid'], how='left')
    candidates_df_chunk = candidates_df_chunk.join(agg_buys_chunk, on=['session', 'candidate_aid'], how='left')
    candidates_df_chunk = candidates_df_chunk.join(agg_buy2buy_chunk, on=['session', 'candidate_aid'], how='left')
    candidates_df_chunk = candidates_df_chunk.join(
        candidates_gnn_chunk.select(['session', 'candidate_aid', 'rank_gnn', 'wgt_gnn']),
        on=['session', 'candidate_aid'],
        how='left'
    )
    
    return candidates_df_chunk

def sample_train(original_train_df: pl.DataFrame,
                 sample_rate: float, 
                 seed: int):
    all_train_sessions = original_train_df['session'].unique()
    sampled_train_sessions = all_train_sessions.sample(fraction=sample_rate, shuffle=True, seed=seed)
    return original_train_df.filter(pl.col('session').is_in(sampled_train_sessions))

def get_history_and_label_df(sampled_train_df: pl.DataFrame):
    session_cutoffs = sampled_train_df.group_by('session').agg(
        (pl.col('ts').max() - 24 * 60 * 60).alias('cutoff_ts')
    )
    
    last_ts_in_train = sampled_train_df['ts'].max()
    _days_ago = last_ts_in_train - (12 * 24 * 60 * 60)
    
    # Chia train_df thành 2 phần
    train_with_cutoffs = sampled_train_df.join(session_cutoffs, on='session', how='left')

    # "Đề bài": Mọi thứ TRƯỚC thời điểm cắt
    history_source_df = train_with_cutoffs.filter((pl.col('ts') < pl.col('cutoff_ts')) & (pl.col('ts') >= _days_ago))

    # "Đáp án": Mọi thứ SAU thời điểm cắt
    labels_source_df = train_with_cutoffs.filter(pl.col('ts') >= pl.col('cutoff_ts'))
    
    train_ground_truth = labels_source_df.group_by(['session', 'type']).agg(pl.col('aid').alias('ground_truth'))
    print("History and Labels for training have been created.")
    
    # Lấy ra tất cả các session ID duy nhất trong giai đoạn này
    unique_sessions_in_recent = history_source_df['session'].unique()

    # Lấy mẫu ngẫu nhiên 50% các session ID này
    # Điều chỉnh `fraction` để có kích thước history_df mong muốn
    sampled_sessions = unique_sessions_in_recent

    # Chỉ giữ lại các sự kiện từ các session đã được lấy mẫu
    history_source_df = history_source_df.filter(pl.col('session').is_in(sampled_sessions))


    # Sắp xếp theo session và thời gian
    history_source_df = history_source_df.sort(['session', 'ts'], descending=[False, True])

    # Chỉ giữ lại 10 tương tác cuối cùng của mỗi session
    # Đây chính là logic "USE TAIL OF SESSION" trong notebook gốc
    #history_df = history_source_df.group_by('session', maintain_order=True).head(30)
    # Bây giờ mới lấy unique
    history_df = history_source_df.select(['session', 'aid']).unique()
    
    return history_df, train_ground_truth, history_source_df

def get_popular_items_df(sampled_train_df: pl.DataFrame):
    session_cutoffs = sampled_train_df.group_by('session').agg(
        (pl.col('ts').max() - 24 * 60 * 60).alias('cutoff_ts')
    )
    
    last_ts_in_train = sampled_train_df['ts'].max()
    _days_ago = last_ts_in_train - (12 * 24 * 60 * 60)
    
    # Chia train_df thành 2 phần
    train_with_cutoffs = sampled_train_df.join(session_cutoffs, on='session', how='left')
    
    recent_train_df = train_with_cutoffs.filter((pl.col('ts') < pl.col('cutoff_ts')) & (pl.col('ts') >= last_ts_in_train - 10*24*60*60))
    # Prepare global popular candidate
    top_clicks_popular = recent_train_df.filter(pl.col('type') == 0)['aid'].value_counts().sort(['count'], descending=[True]).head(15)['aid']
    top_carts_popular = recent_train_df.filter(pl.col('type') == 1)['aid'].value_counts().sort(['count'], descending=[True]).head(20)['aid']
    top_orders_popular = recent_train_df.filter(pl.col('type') == 2)['aid'].value_counts().sort(['count'], descending=[True]).head(20)['aid']

    popular_items = pl.concat([top_clicks_popular, 
                            top_carts_popular, 
                            top_orders_popular]).unique()
    
    popular_items_df = pl.DataFrame({'candidate_aid': popular_items})
    
    return popular_items_df
        
def pre_compute_item_popularity(sampled_train_df: pl.DataFrame):
    item_popularity_df = sampled_train_df['aid'].value_counts().rename({'count': 'popularity'})
    # Chuẩn hóa popularity để tránh số quá lớn (tùy chọn nhưng nên làm)
    min_pop = item_popularity_df['popularity'].quantile(0.05)
    max_pop = item_popularity_df['popularity'].quantile(0.95)
    # --- SỬA LỖI Ở ĐÂY ---
    item_popularity_df = item_popularity_df.with_columns(
        pl.col('popularity').clip(min_pop, max_pop) # Dùng with_columns để sửa đổi cột
    )
    item_popularity_df = item_popularity_df.with_columns(
        ( (pl.col('popularity') + 1 - min_pop) / (max_pop - min_pop) ).alias('popularity_scaled')
    )
    item_popularity_df = item_popularity_df.select(['aid', 'popularity_scaled']).rename({'aid': 'candidate_aid'})
    item_popularity_df = item_popularity_df.filter(pl.col('popularity_scaled') >= 0.15)
    return item_popularity_df
