import gc
import os
import numpy as np
import polars as pl
import tqdm
import lightgbm as lgb
import src.core.config as cfg

def process_chunk(history_chunk: pl.DataFrame, 
                  popular_items_df: pl.DataFrame, # Thêm popular_items vào đây
                  df_clicks: pl.DataFrame, df_buys: pl.DataFrame, df_buy2buy: pl.DataFrame) -> pl.DataFrame:
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
    
    # 4. Tổng hợp tất cả ứng viên cho chunk này
    candidates_df_chunk = pl.concat([
        candidates_history_chunk.select(['session', 'candidate_aid']),
        candidates_popular_chunk, # Thêm nguồn popular ở đây 
        candidates_clicks_chunk.select(['session', 'candidate_aid']),
        candidates_buys_chunk.select(['session', 'candidate_aid']),
        candidates_buy2buy_chunk.select(['session', 'candidate_aid']),
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
    
    return candidates_df_chunk

def sample_train(original_train_df: pl.DataFrame,
                 sample_rate: float, 
                 seed: int):
    all_train_sessions = original_train_df['session'].unique()
    sampled_train_sessions = all_train_sessions.sample(fraction=sample_rate, shuffle=True, seed=seed)
    return original_train_df.filter(pl.col('session').is_in(sampled_train_sessions))

def process_sample_train(sampled_train_df: pl.DataFrame,
                         n_chunks: int,
                         tmp_chunk_path: str,
                         df_clicks_train: pl.DataFrame,
                         df_buys_train: pl.DataFrame,
                         df_buy2buy_train: pl.DataFrame,
                         ):
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
    
    recent_train_df = train_with_cutoffs.filter((pl.col('ts') < pl.col('cutoff_ts')) & (pl.col('ts') >= last_ts_in_train - 10*24*60*60))
    # Prepare global popular candidate
    top_clicks_popular = recent_train_df.filter(pl.col('type') == 0)['aid'].value_counts().sort(['count'], descending=[True]).head(15)['aid']
    top_carts_popular = recent_train_df.filter(pl.col('type') == 1)['aid'].value_counts().sort(['count'], descending=[True]).head(20)['aid']
    top_orders_popular = recent_train_df.filter(pl.col('type') == 2)['aid'].value_counts().sort(['count'], descending=[True]).head(20)['aid']

    popular_items = pl.concat([top_clicks_popular, 
                            top_carts_popular, 
                            top_orders_popular]).unique()

    del train_with_cutoffs, session_cutoffs, unique_sessions_in_recent, sampled_sessions, recent_train_df
    _ = gc.collect()
    
    popular_items_df = pl.DataFrame({'candidate_aid': popular_items})
    
    all_sessions = history_df['session'].unique().to_list()
    chunk_size = len(all_sessions) // n_chunks
    
    os.makedirs(tmp_chunk_path, exist_ok=True)
    
    print(f"\n--- Processing {len(all_sessions)} sessions in {n_chunks+1} chunks ---")
    for i in tqdm(range(n_chunks + 1)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if start >= len(all_sessions):
            break
        
        session_chunk_ids = all_sessions[start:end]
        history_chunk = history_df.filter(pl.col('session').is_in(session_chunk_ids))
        
        # Gọi hàm xử lý cho chunk
        chunk_result = process_chunk(history_chunk, popular_items_df,
                                     df_clicks_train, df_buys_train, df_buy2buy_train)
        # Thêm đặc trưng cuối cùng cho nguồn popular
        chunk_result = chunk_result.with_columns(
            pl.col('candidate_aid').is_in(popular_items_df['candidate_aid']).cast(pl.UInt8).alias('source_popular')
        )
        # --- THAY ĐỔI QUAN TRỌNG: LƯU RA FILE THAY VÌ APPEND VÀO LIST ---
        chunk_result.write_parquet(tmp_chunk_path + f'candidates_chunk_{i}.pqt')
        
        # Dọn dẹp bộ nhớ
        gc.collect()
        
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

def create_training_set_for_type(
    lazy_candidates_df: pl.LazyFrame, 
    all_labels_df: pl.DataFrame, 
    item_popularity_df: pl.DataFrame, # Thêm tham số này
    prediction_type: str, 
    positive_rate: float,
    target_neg_pos_ratio: int, # Ví dụ: 30 có nghĩa là 30 mẫu âm cho mỗi mẫu dương
    popular_fraction: float = 0.5, # Tỷ lệ mẫu âm được lấy từ popular (50%)
    seed: int = 42
) -> pl.DataFrame:
    
    print(f"\n--- Creating training set for '{prediction_type}'  ---")
    prediction_type = cfg.TYPE_LABELS_MAPPING[prediction_type]
    print(prediction_type)
    # 1. Chuẩn bị nhãn (giữ nguyên)
    type_labels = all_labels_df.filter(pl.col('type') == prediction_type) \
                               .explode('ground_truth') \
                               .rename({'ground_truth': 'candidate_aid'}) \
                               .with_columns(pl.lit(1).cast(pl.UInt8).alias('label')) \
                               .select(['session', 'candidate_aid', 'label']) \
                               .unique()

    # 2. Tạo nhãn cho toàn bộ ứng viên (lazy, giữ nguyên)
    labeled_lazy_df = lazy_candidates_df.join(
        type_labels.lazy(),
        on=['session', 'candidate_aid'],
        how='left'
    ).with_columns(pl.col('label').fill_null(0))

    # 1. Tách và thu thập các mẫu dương
    positives_df = labeled_lazy_df.filter(pl.col('label') == 1).collect()
    total_positives = int(len(positives_df) * positive_rate)
    print(f"Collecting positive samples for '{prediction_type}'...")
    if positive_rate < 1.0:
        positives_df = positives_df.sample(total_positives) 
    
    # 2. Thu thập toàn bộ mẫu âm
    print(f"Collecting all negative samples for '{prediction_type}'...")
    negatives_df = labeled_lazy_df.filter(pl.col('label') == 0).collect()

    # 3. Tính toán số lượng N và M
    
    total_negatives_to_sample = int(total_positives * target_neg_pos_ratio)
    
    # Đảm bảo không lấy nhiều hơn số mẫu âm hiện có
    total_negatives_to_sample = min(total_negatives_to_sample, len(negatives_df))
    
    num_popular_negatives = int(total_negatives_to_sample * popular_fraction)
    num_random_negatives = total_negatives_to_sample - num_popular_negatives
    
    print(f"Positives: {total_positives}. Total negatives to sample: {total_negatives_to_sample} ({num_popular_negatives} popular + {num_random_negatives} random)")

    # 4. Lấy mẫu
    # Join với bảng độ phổ biến
    negatives_with_pop = negatives_df.join(item_popularity_df, on='candidate_aid', how='left').fill_null(0)
    
    # --- THAY ĐỔI LỚN Ở ĐÂY ---
    # Lấy N mẫu dựa trên độ phổ biến theo cách của Polars
    # Thêm một cột số ngẫu nhiên và chia cho trọng số, sau đó lấy top N
    
    # Thêm một epsilon nhỏ để tránh chia cho 0
    epsilon = 1e-6 
    # Bước 4.1: Lấy N mẫu dựa trên độ phổ biến
    print("Sampling popular negatives...")
    
    # Lấy ra cột trọng số dưới dạng một Series
    weights = negatives_with_pop['popularity_scaled']
    
    # Tạo ra các giá trị ngẫu nhiên có trọng số
    random_weighted_values = np.random.rand(len(negatives_with_pop)) ** (1 / (weights.to_numpy() + epsilon))
    
    # Thêm cột này vào DataFrame
    negatives_with_pop = negatives_with_pop.with_columns(
        pl.Series("random_weighted", random_weighted_values)
    )
    
    # Lấy top K dựa trên cột vừa tạo
    popular_negatives_df = negatives_with_pop.top_k(k=num_popular_negatives, by='random_weighted').drop('random_weighted')
    
    # Bước 4.2: Lấy M mẫu ngẫu nhiên từ phần còn lại
    print("Sampling random negatives...")
    remaining_negatives = negatives_with_pop.join(
        popular_negatives_df.select(['session', 'candidate_aid']),
        on=['session', 'candidate_aid'], 
        how='anti'
    )
    random_negatives_df = remaining_negatives.sample(
        n=num_random_negatives, 
        shuffle=True, 
        seed=seed
    )
    
    # 5. Gộp lại
    final_cols = positives_df.columns
    final_df = pl.concat([
        positives_df,
        popular_negatives_df.select(final_cols),
        random_negatives_df.select(final_cols)
    ])
    
    return final_df

def create_time_window_features(
    feature_source_df: pl.DataFrame, 
    time_window_days: int = None
) -> pl.DataFrame:
    """
    Hàm con để tính toán các feature item trên một cửa sổ thời gian cụ thể.
    """
    if time_window_days is not None:
        last_ts = feature_source_df['ts'].max()
        start_ts = last_ts - (time_window_days * 24 * 60 * 60)
        source_df = feature_source_df.filter(pl.col('ts') >= start_ts)
        suffix = f'_{time_window_days}d'
    else:
        source_df = feature_source_df
        suffix = '_all'
        
    item_feats = source_df.group_by('aid').agg([
        pl.count().alias(f'item_total_counts{suffix}'),
        pl.col('type').filter(pl.col('type') == 0).count().alias(f'item_click_counts{suffix}'),
        pl.col('type').filter(pl.col('type') == 1).count().alias(f'item_cart_counts{suffix}'),
        pl.col('type').filter(pl.col('type') == 2).count().alias(f'item_order_counts{suffix}'),
    ]).rename({'aid': 'candidate_aid'})
    
    # Tính toán tỷ lệ chuyển đổi, xử lý chia cho 0
    item_feats = item_feats.with_columns([
        (pl.col(f'item_order_counts{suffix}') / (pl.col(f'item_click_counts{suffix}') + 1e-6)).alias(f'item_buy_ratio_c{suffix}'),
        (pl.col(f'item_cart_counts{suffix}') / (pl.col(f'item_click_counts{suffix}') + 1e-6)).alias(f'item_cart_ratio_c{suffix}'),
    ])
    
    return item_feats

def create_session_feat_feature(session_context_df: pl.DataFrame):
    return session_context_df.group_by('session').agg([
        pl.count().alias('session_length'),
        pl.col('aid').n_unique().alias('session_unique_aids'),
        pl.col('ts').max().alias('session_end_ts')
    ])

def create_repetion_feature(session_context_df: pl.DataFrame):
    return session_context_df.group_by(['session', 'aid']).agg(
        pl.count().alias('num_repetitions_in_session')
    ).rename({'aid': 'candidate_aid'})

def create_last_items_ts_feature(session_context_df: pl.DataFrame):
    return session_context_df.group_by(['session', 'aid']).agg(
        pl.col('ts').max().alias('last_item_ts')
    ).rename({'aid': 'candidate_aid'})

def create_last_items_feature(session_context_df: pl.DataFrame):
    last_event_in_session = session_context_df.sort('ts').group_by('session', maintain_order=True).tail(1)
    last_item_feats = last_event_in_session.select(['session', 'aid']).rename({'aid': 'candidate_aid'})
    last_item_feats = last_item_feats.with_columns(pl.lit(1).cast(pl.UInt8).alias('is_last_item'))
    return last_item_feats

def create_session_pacing_feature(session_context_df: pl.DataFrame):
    time_diffs = session_context_df.sort('ts').select([
        pl.col('session'),
        pl.col('ts').diff().over('session').alias('time_diff_from_previous')
    ])
    session_pacing_feats = time_diffs.group_by('session').agg([
        pl.col('time_diff_from_previous').mean().alias('session_time_diff_mean'),
        pl.col('time_diff_from_previous').std().alias('session_time_diff_std'),
        pl.col('time_diff_from_previous').max().alias('session_time_diff_max'),
        pl.col('time_diff_from_previous').min().alias('session_time_diff_min'),
    ])
    return session_pacing_feats

def add_features(df: pl.DataFrame, 
                 time_window_feats: list[pl.DataFrame],
                 session_feats: pl.DataFrame,
                 repetition_feats: pl.DataFrame,
                 last_item_feats: pl.DataFrame,
                 session_pacing_feats: pl.DataFrame,
                 last_item_ts_in_session: pl.DataFrame) -> pl.DataFrame:
   
    print("  - Joining all new features...")
    for tw_feat in time_window_feats:
        df = df.join(tw_feat, on='candidate_aid', how='left')
    df = df.join(session_feats, on='session', how='left')
    df = df.join(repetition_feats, on=['session', 'candidate_aid'], how='left') # Join feature lặp lại
    df = df.join(last_item_feats, on=['session', 'candidate_aid'], how='left') # Join feature item cuối
    df = df.join(session_pacing_feats, on='session', how='left') # Join feature nhịp độ
    df = df.join(last_item_ts_in_session, on=['session', 'candidate_aid'], how='left')
    
    # --- 1.5: Tính toán các đặc trưng dẫn xuất và dọn dẹp ---
    df = df.with_columns(
        (pl.col('session_end_ts') - pl.col('last_item_ts')).alias('time_since_last_seen')
    )
    
    # Điền các giá trị null
    df = df.with_columns(
        pl.col('time_since_last_seen').fill_null(30 * 24 * 60 * 60) # Điền giá trị lớn
    )
    df = df.fill_null(0) # Điền 0 cho tất cả các null còn lại
    
    # Bỏ các cột timestamp không cần thiết nữa
    df = df.drop(['session_end_ts', 'last_item_ts'])
    
    return df

def train_lgbm_model(df: pl.DataFrame, model_type: str) -> lgb.LGBMRanker:
    """
    Chuẩn bị dữ liệu và huấn luyện một mô hình LGBMRanker.
    """
    print(f"--- Step 2&3: Training LGBMRanker for '{model_type}' ---")
    
    # --- 2.1: Chuẩn bị X, y, groups ---
    # Loại bỏ các cột không phải là feature
    non_feature_cols = ['session', 'candidate_aid', 'label']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    print(f"  - Using {len(feature_cols)} features.")

    # Sắp xếp theo session để đảm bảo 'groups' được tính đúng
    df = df.sort('session')
    
    X_train = df.select(feature_cols).to_numpy()
    y_train = df.select('label').to_numpy().ravel()
    
    # Tính toán group array: số lượng ứng viên cho mỗi session
    groups = df.group_by('session', maintain_order=True).count()['count'].to_numpy()
    
    # --- 2.2: Huấn luyện mô hình ---
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    
    print("  - Fitting model...")
    model.fit(
        X_train,
        y_train,
        group=groups,
        eval_set=[(X_train, y_train)],
        eval_group=[groups],
        callbacks=[lgb.early_stopping(10, verbose=False)]
    )
    
    # Lưu lại mô hình
    model.booster_.save_model(f'lgbm_ranker_{model_type}.txt')
    print(f"  - Model for '{model_type}' saved.")
    
    return model