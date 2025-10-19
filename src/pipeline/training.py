from typing import Any, Dict
import numpy as np
import polars as pl
import lightgbm as lgb
import src.core.config as cfg

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

def train_lgbm_model(df: pl.DataFrame, model_type: str,
                     model_params: Dict[str, Any] | None = None) -> lgb.LGBMRanker:
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
    if model_params is None:
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="map",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = lgb.LGBMRanker(**model_params)
    
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
