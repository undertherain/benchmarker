def estimate_attention_gflop_per_sample(len_seq, embed_dim):
    cnt_projections = 4
    ops_proj = 2 * len_seq * embed_dim * embed_dim * cnt_projections
    ops_Q_K = 2 * len_seq * len_seq * embed_dim
    ops_Q_Kt_V = ops_Q_K
    ops_estimated = ((ops_proj + ops_Q_K + ops_Q_Kt_V)) / (10 ** 9)
    return ops_estimated
