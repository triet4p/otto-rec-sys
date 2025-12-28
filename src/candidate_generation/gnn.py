import numpy as np
import polars as pl
import faiss
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

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
    
class BPRDataset(Dataset):
    def __init__(self, interaction_pairs, all_item_indices, user_pos_items_dict, num_aids):
        """
        Khởi tạo Dataset cho BPR Loss.
        
        Args:
            interaction_pairs (list): List các tuple (user_idx, positive_item_idx).
            all_item_indices (set): Set chứa tất cả các item index.
            user_pos_items_dict (dict): Dict: user_idx -> set(positive_item_indices).
            num_aids (int): Tổng số lượng item.
        """
        self.interaction_pairs = interaction_pairs
        self.all_item_indices = list(all_item_indices) # Chuyển sang list để có thể index
        self.user_pos_items = user_pos_items_dict
        self.num_aids = num_aids

    def __len__(self):
        return len(self.interaction_pairs)

    def __getitem__(self, index):
        # 1. Lấy một cặp tương tác dương (user, positive_item)
        user_idx, pos_item_idx = self.interaction_pairs[index]
        
        # 2. Lấy mẫu một item âm (negative_item)
        neg_item_idx = None
        while neg_item_idx is None or neg_item_idx in self.user_pos_items[user_idx]:
            # Lấy ngẫu nhiên một item từ toàn bộ danh sách
            neg_item_idx = random.choice(self.all_item_indices)
            
        return user_idx, pos_item_idx, neg_item_idx
    
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Chúng ta vẫn cần khởi tạo các lớp GCNConv,
        # mặc dù chỉ dùng phương thức propagate của chúng.
        self.convs = nn.ModuleList([GCNConv(embed_dim, embed_dim) for _ in range(num_layers)])

    def forward(self, edge_index):
        # Lấy embedding khởi tạo
        x0 = self.embedding.weight
        all_layer_embeddings = [x0]
        
        # Chuẩn hóa ma trận kề một lần duy nhất
        # Đây là bước quan trọng nhất để mô phỏng LightGCN
        norm_edge_index, norm_edge_weight = gcn_norm(
            edge_index, 
            num_nodes=self.num_users + self.num_items
        )

        x = x0
        for conv in self.convs:
            # GỌI TRỰC TIẾP `propagate` ĐỂ THỰC HIỆN PHÉP TỔNG HỢP HÀNG XÓM
            # mà không cần qua lớp Linear `conv.lin`
            x = conv.propagate(norm_edge_index, x=x, edge_weight=norm_edge_weight)
            all_layer_embeddings.append(x)
        
        final_embedding = torch.mean(torch.stack(all_layer_embeddings, dim=0), dim=0)
        
        users_emb, items_emb = torch.split(final_embedding, [self.num_users, self.num_items])
        
        return users_emb, items_emb
    
def bpr_loss(users_emb, pos_items_emb, neg_items_emb):
    # Đảm bảo các tensor có ít nhất 2 chiều
    #print(users_emb.dim(), users_emb)
    if users_emb.dim() == 1: users_emb = users_emb.unsqueeze(0)
    if pos_items_emb.dim() == 1: pos_items_emb = pos_items_emb.unsqueeze(0)
    if neg_items_emb.dim() == 1: neg_items_emb = neg_items_emb.unsqueeze(0)
        
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores)) # Dùng F.logsigmoid ổn định hơn