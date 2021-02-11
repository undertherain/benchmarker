def estimate_attention_gflop_per_sample(len_seq, embed_dim):
    cnt_projections = 4  # q, k, v, out
    macs_proj = len_seq * embed_dim * embed_dim * cnt_projections
    macs_Q_K = len_seq * len_seq * embed_dim
    macs_Q_Kt_V = macs_Q_K
    ops_in_mac = 2
    return ops_in_mac * (macs_proj + macs_Q_K + macs_Q_Kt_V) / (10 ** 9)
