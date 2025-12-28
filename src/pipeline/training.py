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

def create_time_window_features(feature_source_df: pl.DataFrame, time_window_days: int = None) -> pl.DataFrame:
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
    
    # Tính tỷ lệ chuyển đổi (Smoothing +10 để tránh nhiễu ở item ít tương tác)
    item_feats = item_feats.with_columns([
        (pl.col(f'item_order_counts{suffix}') / (pl.col(f'item_click_counts{suffix}') + 10)).alias(f'item_buy_ratio{suffix}'),
        (pl.col(f'item_cart_counts{suffix}') / (pl.col(f'item_click_counts{suffix}') + 10)).alias(f'item_cart_ratio{suffix}'),
    ])
    return item_feats

# --- 2. Feature Động (Session Context) ---
def create_session_context_features(session_context_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Tính toán tất cả các feature dựa trên context của session.
    Trả về: (session_level_features, interaction_level_features, last_item_info)
    """
    # A. Session Level Features
    session_feats = session_context_df.group_by('session').agg([
        pl.count().alias('session_length'),
        pl.col('aid').n_unique().alias('session_unique_aids'),
        pl.col('ts').max().alias('session_end_ts'),
        (pl.col('ts').max() - pl.col('ts').min()).alias('session_duration'),
    ])
    
    # B. Interaction Level Features (Lặp lại & Thời gian cuối)
    interaction_feats = session_context_df.group_by(['session', 'aid']).agg([
        pl.count().alias('num_repetitions'),
        pl.col('ts').max().alias('last_item_ts')
    ]).rename({'aid': 'candidate_aid'})
    
    # C. Last Item Info (Item cuối cùng user xem)
    last_items = session_context_df.sort('ts').group_by('session', maintain_order=True).last()
    last_items = last_items.select(['session', 'aid']).rename({'aid': 'last_aid'})
    
    return session_feats, interaction_feats, last_items

def add_sorted_rank_features(df: pl.DataFrame, fill_value: int = 999) -> pl.DataFrame:
    """Tạo feature rank tốt nhất từ tất cả các nguồn."""
    rank_cols = [col for col in df.columns if col.startswith('rank_')]
    if not rank_cols: return df

    # Tạo list rank, sort và lấy ra các giá trị min
    rank_exprs = [pl.col(c).fill_null(fill_value) for c in rank_cols]
    
    df = df.with_columns(
        pl.concat_list(rank_exprs).list.sort().alias('temp_sorted_ranks')
    )
    
    # Tách ra min_rank_1 (best), min_rank_2 (2nd best)
    new_cols = [
        pl.col('temp_sorted_ranks').list.get(0).alias('min_rank_1'),
        pl.col('temp_sorted_ranks').list.get(1).alias('min_rank_2'),
        # Số lượng nguồn gợi ý item này
        (pl.col('temp_sorted_ranks').list.eval(pl.element() < fill_value).list.sum()).alias('n_sources_present')
    ]
    return df.with_columns(new_cols).drop('temp_sorted_ranks')

def add_features(df: pl.DataFrame, 
                 time_window_feats_list: list[pl.DataFrame], # [item_all, item_7d]
                 session_feats: pl.DataFrame,
                 interaction_feats: pl.DataFrame,
                 last_items: pl.DataFrame) -> pl.DataFrame:
   
    # 1. Join các bảng gốc (Giữ nguyên)
    for tw_feat in time_window_feats_list:
        df = df.join(tw_feat, on='candidate_aid', how='left')
    df = df.join(session_feats, on='session', how='left')
    df = df.join(interaction_feats, on=['session', 'candidate_aid'], how='left')
    df = df.join(last_items, on='session', how='left')

    # --- NHÓM 1: TREND / VELOCITY (Tốc độ tăng trưởng của Item) ---
    # So sánh 7 ngày vs All time. 
    # Logic: Item có tỷ trọng click trong 7 ngày cao bất thường so với lịch sử -> Đang Hot.
    
    # Giả sử time_window_feats_list[0] là ALL, [1] là 7D
    # Các cột sẽ có suffix '_all' và '_7d'
    
    df = df.with_columns([
        # Tỷ lệ click gần đây / click tổng (Cộng 10 để tránh chia 0 và nhiễu)
        (pl.col('item_click_counts_7d').fill_null(0) / (pl.col('item_click_counts_all').fill_null(0) + 10)).alias('click_trend_7d_vs_all'),
        
        # Tỷ lệ order gần đây / order tổng
        (pl.col('item_order_counts_7d').fill_null(0) / (pl.col('item_order_counts_all').fill_null(0) + 10)).alias('order_trend_7d_vs_all'),
        
        # Conversion Rate thay đổi thế nào? (CR 7 ngày - CR All)
        (pl.col('item_buy_ratio_7d').fill_null(0) - pl.col('item_buy_ratio_all').fill_null(0)).alias('conversion_trend_diff')
    ])

    # --- NHÓM 2: CROSS-SOURCE COMPARISON (So sánh giữa các nguồn Co-visit) ---
    # Logic: Sự chênh lệch thứ hạng giữa các nguồn nói lên điều gì?
    # Ví dụ: Rank Buy2Buy thấp (tốt) nhưng Rank Clicks cao (tệ) -> Item này ít người click nhưng hễ click là mua -> Tiềm năng cao.
    
    # Fill null rank bằng 999 trước khi tính toán
    rank_cols = ['rank_clicks', 'rank_buys', 'rank_buy2buy']
    for c in rank_cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(999).alias(c))
        else:
            df = df.with_columns(pl.col(c).fill_null(999))

    df = df.with_columns([
        # Chênh lệch rank
        (pl.col('rank_clicks') - pl.col('rank_buy2buy')).alias('rank_diff_click_buy2buy'),
        (pl.col('rank_buys') - pl.col('rank_buy2buy')).alias('rank_diff_buys_buy2buy'),
        
        # Tổng hợp trọng số (Weighted Sum) - Tạo ra một "Siêu điểm số"
        (pl.col('wgt_buy2buy').fill_null(0) * 2 + pl.col('wgt_buys').fill_null(0) * 1).alias('combined_buy_weight')
    ])

    # --- NHÓM 3: CONTEXTUAL RECENCY (Tính gần đây kết hợp ngữ cảnh) ---
    # Logic: Recency quan trọng, nhưng Recency của một item "Hot" quan trọng hơn Recency của item "Rác".
    
    # Tính Recency cơ bản trước
    df = df.with_columns(
        (pl.col('session_end_ts') - pl.col('last_item_ts')).fill_null(7*24*3600).alias('recency_score')
    )
    
    # Log Recency để giảm biên độ số (giây -> log giây)
    df = df.with_columns(
        pl.col('recency_score').log1p().alias('log_recency_score')
    )
    
    # Tương tác: Điểm Co-visit chia cho thời gian (Càng gần càng giá trị)
    # Thêm 1 vào log_recency để tránh chia 0
    df = df.with_columns([
        (pl.col('wgt_buy2buy').fill_null(0) / (pl.col('log_recency_score') + 1)).alias('wgt_buy2buy_decayed'),
        (pl.col('wgt_clicks').fill_null(0) / (pl.col('log_recency_score') + 1)).alias('wgt_clicks_decayed')
    ])

    # --- NHÓM 4: CÁC CỜ (FLAGS) QUAN TRỌNG (Giữ lại từ cũ) ---
    df = df.with_columns([
        (pl.col('candidate_aid') == pl.col('last_aid')).cast(pl.Int8).fill_null(0).alias('is_last_viewed'),
        # Item này có phải là item phổ biến nhất trong session không? (Logic đơn giản: count > 1)
        (pl.col('num_repetitions') > 1).cast(pl.Int8).alias('is_repeated_in_session')
    ])

    # --- Sorted Ranks (Giữ nguyên - rất mạnh) ---
    df = add_sorted_rank_features(df)

    # Dọn dẹp
    df = df.fill_null(0)
    cols_to_drop = ['session_end_ts', 'last_item_ts', 'last_aid', 'session_duration', 'first_item_ts'] 
    df = df.drop([c for c in cols_to_drop if c in df.columns])
    
    return df

def select_best_features(df: pl.DataFrame, target_type: str, top_k: int = 60):
    """
    Huấn luyện nhanh 1 model để chọn ra top_k features tốt nhất cho target_type.
    """
    print(f"  >> Performing Feature Selection for {target_type}...")
    
    ignore_cols = ['session', 'candidate_aid', 'label']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Sample dữ liệu để chạy nhanh (ví dụ 2 triệu dòng)
    if len(df) > 2_000_000:
        df_sample = df.sample(n=2_000_000, seed=42)
    else:
        df_sample = df
        
    X = df_sample.select(feature_cols).to_numpy()
    y = df_sample.select('label').to_numpy().ravel()
    groups = df_sample.group_by('session', maintain_order=True).len()['len'].to_numpy()
    
    # Train model nhẹ
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="map",
        n_estimators=50, learning_rate=0.1, max_depth=5,
        importance_type='gain', random_state=42, n_jobs=-1
    )
    model.fit(X, y, group=groups)
    
    # Lấy feature importance
    imp_df = pl.DataFrame({
        'feature': feature_cols,
        'gain': model.feature_importances_
    }).sort('gain', descending=True)
    
    # Chọn top K
    best_feats = imp_df.head(top_k)['feature'].to_list()
    print(f"     Selected {len(best_feats)} features. Top 5: {best_feats[:5]}")
    
    # (Tùy chọn) In ra các feature bị loại bỏ để kiểm tra
    # dropped = [f for f in feature_cols if f not in best_feats]
    # print(f"     Dropped: {dropped[:5]}...")
    
    return best_feats

def train_final_model(df: pl.DataFrame, features: list, model_type: str):
    """Huấn luyện model chính thức với danh sách feature đã chọn."""
    print(f"  >> Training Final Model for {model_type} with {len(features)} features...")
    
    df = df.sort('session')
    X = df.select(features).to_numpy()
    y = df.select('label').to_numpy().ravel()
    groups = df.group_by('session', maintain_order=True).len()['len'].to_numpy()
    
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="map",
        n_estimators=500, learning_rate=0.05, num_leaves=32,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1
    )
    
    # --- SỬA LỖI Ở ĐÂY ---
    # Thêm eval_set và eval_group
    model.fit(
        X, 
        y, 
        group=groups, 
        eval_set=[(X, y)],       # Đưa tập train vào làm tập đánh giá
        eval_group=[groups],     # Cung cấp thông tin group cho tập đánh giá
        callbacks=[lgb.early_stopping(50, verbose=False)] # Tăng patience lên 50 cho an toàn
    )
    return model