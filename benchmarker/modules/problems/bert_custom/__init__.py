from benchmarker.modules.problems.attention.pytorch import \
    estimate_attention_gflop_per_sample


def estimate_gflop_per_sample(len_seq, embed_dim, lin_dim, nb_layers):
    attention_gflop = estimate_attention_gflop_per_sample(len_seq, embed_dim)
    mid_linear_gflop = 2 * embed_dim * lin_dim * len_seq / (10 ** 9)
    top_linear_gflop = 2 * embed_dim * lin_dim * len_seq / (10 ** 9)
    layer_gflop = attention_gflop + mid_linear_gflop + top_linear_gflop
    # TODO: add layer norm
    return nb_layers * layer_gflop
