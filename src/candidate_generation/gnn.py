import numpy as np
import polars as pl
import faiss

def generate_candidates_from_gnn(
    history_chunk: pl.DataFrame,
    embedding_df: pl.DataFrame, # DataFrame Polars chứa embedding
    faiss_index: faiss.Index,
    idx2aid_faiss: dict,
    top_k: int = 50
) -> pl.DataFrame:
    """
    Tạo ứng viên bằng cách lấy trung bình embedding của session và tìm kiếm trên FAISS.
    """
    history_with_emb = history_chunk.join(embedding_df, on='aid', how='inner')

    if len(history_with_emb) == 0:
        return pl.DataFrame(schema={'session': pl.Int32, 'candidate_aid': pl.Int32, 'rank_gnn': pl.UInt32, 'wgt_gnn': pl.Float32}) # Trả về schema đúng

    # --- SỬA LỖI Ở ĐÂY: TÍNH TRUNG BÌNH AN TOÀN ---
    # 1. "Trải phẳng" các vector embedding
    exploded_emb = history_with_emb.explode('embedding')
    
    # 2. Thêm một cột index cho mỗi phần tử của vector
    # Điều này cần thiết để có thể nhóm chúng lại sau này
    exploded_emb = exploded_emb.with_columns(
        pl.arange(0, pl.len()).over('session', 'aid').alias('emb_idx')
    )
    
    # 3. Tính trung bình cho mỗi vị trí index của vector
    mean_emb_parts = exploded_emb.group_by('session', 'emb_idx').agg(
        pl.col('embedding').mean()
    ).sort(['session', 'emb_idx'])
    
    # 4. Gom các phần trung bình lại thành vector session cuối cùng
    session_embeddings = mean_emb_parts.group_by('session', maintain_order=True).agg(
        pl.col('embedding').alias('session_embedding')
    )

    # --- CÁC BƯỚC CÒN LẠI GIỮ NGUYÊN ---
    # (Phần code xây dựng query_vectors từ session_embeddings, 
    # tìm kiếm FAISS, và xử lý kết quả không thay đổi)
    
    num_sessions_in_chunk = len(session_embeddings)
    embed_size = len(session_embeddings[0, 'session_embedding'])
    query_vectors = np.zeros((num_sessions_in_chunk, embed_size), dtype=np.float32)
    for i, row in enumerate(session_embeddings.iter_rows()):
        query_vectors[i, :] = row[1]

    faiss.normalize_L2(query_vectors)
    
    distances, indices = faiss_index.search(query_vectors, top_k)
    
    # 4. Xử lý kết quả thành DataFrame
    sessions = []
    candidates = []
    ranks = []
    scores = []
    
    session_ids = session_embeddings['session'].to_list()
    for i in range(len(session_ids)):
        for j in range(top_k):
            sessions.append(session_ids[i])
            candidates.append(idx2aid_faiss[indices[i, j]])
            ranks.append(j)
            scores.append(distances[i, j])
            
    return pl.DataFrame({
        'session': sessions,
        'candidate_aid': candidates,
        'rank_gnn': ranks,
        'wgt_gnn': scores
    })