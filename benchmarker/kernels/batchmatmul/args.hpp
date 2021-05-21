// traditional BLAS API:
// M - Number of rows in matrices A and C.
// N - Number of columns in matrices B and C.
// K - Number of columns in matrix A; number of rows in matrix B.
// we remap from intuitive order to BLAS-style

struct Options {
    std::string precision;
    size_t cnt_rows_A_rows_C;
    size_t cnt_cols_A_rows_B;
    size_t cnt_cols_B_cols_C;
    size_t batch_size;
    size_t nb_epoch;
};

Options parse_args(const int argc, char *argv[]) {
    if (argc != 7)
    {
        std::cerr << "provide precision, m, n, k, batch_size, nb_epoch as command line parameters\n";
        std::cerr << "got " << argc << " parameters\n";
        exit(-1);
    }
    Options options;
    options.precision = std::string(argv[1]);
    options.cnt_rows_A_rows_C = atoi(argv[2]);
    options.cnt_cols_A_rows_B = atoi(argv[3]);
    options.cnt_cols_B_cols_C = atoi(argv[4]);
    options.batch_size = atoi(argv[5]);
    options.nb_epoch = atoi(argv[6]);
    return options;
}

template<typename precision>
void get_matrices(size_t &cnt_rows_A_rows_C,
                  size_t &cnt_cols_A_rows_B,
                  size_t &cnt_cols_B_cols_C,
                  precision * & A,
                  precision * & B,
                  precision * & C) {
    size_t i;
    A = (precision*) malloc(sizeof(precision) * cnt_rows_A_rows_C * cnt_cols_A_rows_B);
    B = (precision*) malloc(sizeof(precision) * cnt_cols_A_rows_B * cnt_cols_B_cols_C);
    C = (precision*) malloc(sizeof(precision) * cnt_rows_A_rows_C * cnt_cols_B_cols_C);
    //fprintf(stderr, "done malloc\n");
    for(i=0; i < cnt_rows_A_rows_C * cnt_cols_A_rows_B; i++) { A[i] = static_cast<precision>(rand())/RAND_MAX;}
    for(i=0; i < cnt_cols_A_rows_B * cnt_cols_B_cols_C; i++) { B[i] = static_cast<precision>(rand())/RAND_MAX;}
    for(i=0; i < cnt_rows_A_rows_C * cnt_cols_B_cols_C; i++) { C[i] = static_cast<precision>(rand())/RAND_MAX;}
    //fprintf(stderr, "done random init\n");
}

template<typename precision>
void get_batched_matrices(size_t &cnt_rows_A_rows_C,
                          size_t &cnt_cols_A_rows_B,
                          size_t &cnt_cols_B_cols_C,
                          precision ** & A,
                          precision ** & B,
                          precision ** & C,
                          size_t batch_size) {
    A = (precision**) malloc(sizeof(precision*) * batch_size);
    B = (precision**) malloc(sizeof(precision*) * batch_size);
    C = (precision**) malloc(sizeof(precision*) * batch_size);
    fprintf(stderr, "done malloc\n");
    for(size_t i=0; i<batch_size; i++){
        get_matrices<precision>(cnt_rows_A_rows_C, cnt_cols_A_rows_B, cnt_cols_B_cols_C,
                                A[i], B[i], C[i]);
    }
    fprintf(stderr, "done random init\n");
}