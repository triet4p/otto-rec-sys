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
    # 1. Lấy embedding cho tất cả các aid trong chunk lịch sử
    history_with_emb = history_chunk.join(embedding_df.select(['aid', 'embedding']), on='aid', how='inner')

    # 2. Tính session embedding trung bình
    session_embeddings = history_with_emb.group_by('session').agg(
        pl.mean('embedding').alias('session_embedding')
    ).sort('session')

    if len(session_embeddings) == 0:
        return pl.DataFrame(schema={'session': pl.Int64, 'candidate_aid': pl.Int32, 'rank_gnn': pl.UInt16, 'wgt_gnn': pl.Float32})

    # 3. Chuẩn bị vector truy vấn và tìm kiếm
    num_sessions_in_chunk = len(session_embeddings)
    embed_size = len(session_embeddings[0, 'session_embedding'])
    
    # Khởi tạo một mảng NumPy 2D rỗng
    query_vectors = np.zeros((num_sessions_in_chunk, embed_size), dtype=np.float32)
    
    # Lặp và điền vào mảng
    for i, row in enumerate(session_embeddings.iter_rows()):
        # row[1] chính là list embedding
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