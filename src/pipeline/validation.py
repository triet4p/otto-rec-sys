import polars as pl
import pandas as pd
from typing import Literal


def calculate_localcv(model_type: Literal['clicks', 'carts', 'orders'],
                      predictions_df: pl.DataFrame,
                      ground_truth_pd: pd.DataFrame) -> float:
    print(f"\n--- Processing for type: {model_type} ---")
    
    # 1. Lấy ra các cột cần thiết cho việc xếp hạng
    score_col = f'score_{model_type}'
    preds_for_type = predictions_df.select(['session', 'candidate_aid', score_col])
    
    # 2. Sắp xếp các ứng viên theo điểm số và lấy top 20 cho mỗi session
    print("  - Ranking and getting top 20...")
    top_20_preds = preds_for_type.sort(score_col, descending=True) \
                                 .group_by('session', maintain_order=False) \
                                 .head(20) \
                                 .group_by('session', maintain_order=True) \
                                 .agg(pl.col('candidate_aid').alias('labels'))
    
    # 3. Chuyển "bài làm" (predictions) sang Pandas để dễ dàng merge và tính toán
    top_20_preds_pd = top_20_preds.to_pandas()
    
    # 4. Lấy "đáp án" (ground truth) cho loại hiện tại
    gt_pd = ground_truth_pd[ground_truth_pd['type'] == model_type].copy()
    
    # 5. Merge "bài làm" và "đáp án"
    print("  - Merging predictions with ground truth...")
    # Dùng left join để đảm bảo giữ lại tất cả các session có trong ground truth
    merged_df = gt_pd.merge(top_20_preds_pd, on='session', how='left')
    
    # Xử lý các session trong ground truth mà chúng ta không có dự đoán nào
    # (điều này hiếm khi xảy ra nhưng là một bước an toàn)
    merged_df['labels'] = merged_df['labels'].fillna("").apply(list)
    
    # 6. Tính toán số lần đoán trúng (hits)
    print("  - Calculating hits...")
    # Dùng list comprehension và set intersection để tăng tốc
    hits = [len(set(gt).intersection(set(pred))) for gt, pred in zip(merged_df['ground_truth'], merged_df['labels'])]
    merged_df['hits'] = hits
    
    # Đếm số lượng đáp án thật (giới hạn ở 20)
    merged_df['gt_count'] = merged_df['ground_truth'].str.len().clip(0, 20)
    
    # 7. Tính recall và điểm tổng
    recall = merged_df['hits'].sum() / merged_df['gt_count'].sum()
    
    print(f"  ==> {model_type} recall = {recall:.5f}")
    
    return recall
    