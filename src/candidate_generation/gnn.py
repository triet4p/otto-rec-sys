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
    print("\n--- DEBUG: Inside generate_candidates_from_gnn ---")
    
    # --- DEBUG STEP 1: Kiểm tra đầu vào ---
    print(f"  - history_chunk shape: {history_chunk.shape}")
    if len(history_chunk) == 0:
        print("  - DEBUG: history_chunk is empty. Exiting early.")
        # Trả về schema đúng để không gây lỗi ở các bước sau
        return pl.DataFrame(schema={'session': pl.Int64, 'candidate_aid': pl.Int32, 'rank_gnn': pl.UInt32, 'wgt_gnn': pl.Float32})

    # --- Bước join ---
    history_with_emb = history_chunk.join(embedding_df, on='aid', how='inner')
    
    # --- DEBUG STEP 2: Kiểm tra kết quả của phép join ---
    print(f"  - history_with_emb shape after join: {history_with_emb.shape}")
    if len(history_with_emb) == 0:
        print("  - DEBUG: No AIDs from history_chunk were found in embedding_df. Exiting early.")
        return pl.DataFrame(schema={'session': pl.Int64, 'candidate_aid': pl.Int32, 'rank_gnn': pl.UInt32, 'wgt_gnn': pl.Float32})

    # --- Bước group by ---
    session_embeddings = history_with_emb.group_by('session').agg(
        pl.mean('embedding').alias('session_embedding')
    ).sort('session')

    # --- DEBUG STEP 3: Kiểm tra kết quả của phép group_by ---
    print(f"  - session_embeddings shape after group_by: {session_embeddings.shape}")
    print(f"  - Schema of session_embeddings: {session_embeddings.schema}")
    
    if len(session_embeddings) == 0:
        print("  - DEBUG: group_by resulted in an empty DataFrame. Exiting early.")
        return pl.DataFrame(schema={'session': pl.Int64, 'candidate_aid': pl.Int32, 'rank_gnn': pl.UInt32, 'wgt_gnn': pl.Float32})
        
    # --- DEBUG STEP 4: Kiểm tra trực tiếp giá trị gây lỗi ---
    first_row_embedding = session_embeddings[0, 'session_embedding']
    print(f"  - Value of 'session_embedding' in the first row: {first_row_embedding}")
    if first_row_embedding is None:
        print("  - ❌ DEBUG: FOUND THE CULPRIT! The value is None.")
        # Vẫn tiếp tục để gây ra lỗi gốc, nhưng chúng ta đã biết nguyên nhân
    else:
        print(f"  - Type of the value: {type(first_row_embedding)}")


    # --- Dòng code gây lỗi gốc ---
    try:
        num_sessions_in_chunk = len(session_embeddings)
        embed_size = len(session_embeddings[0, 'session_embedding'])

        query_vectors = np.zeros((num_sessions_in_chunk, embed_size), dtype=np.float32)
        for i, row in enumerate(session_embeddings.iter_rows()):
            query_vectors[i, :] = row[1]

        # ... (phần còn lại của hàm) ...

    except TypeError as e:
        print("\n  - ❌ DEBUG: TypeError was caught exactly as expected.")
        print(f"  - Error message: {e}")
        print("  - This confirms that the value at session_embeddings[0, 'session_embedding'] is None.")
        # Ném lại lỗi để dừng chương trình
        raise e
        
    # Nếu code chạy đến đây mà không có lỗi, có nghĩa là vấn đề nằm ở đâu đó khác
    print("  - DEBUG: Passed the error-prone line without TypeError. The issue might be different.")
        
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