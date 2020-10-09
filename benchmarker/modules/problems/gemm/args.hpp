// traditional BLAS API:
// M - Number of rows in matrices A and C.
// N - Number of columns in matrices B and C.
// K - Number of columns in matrix A; number of rows in matrix B.
// we remap from intuitive order to BLAS-style

void parse_args(const int argc,
                char *argv[],
                std::string &precision,
                size_t &cnt_rows_A_rows_C,
                size_t &cnt_cols_A_rows_B,
                size_t &cnt_cols_B_cols_C
                ) {
    if ((argc != 3) && (argc != 5))
    {
        std::cerr << "provide precision, m, n, k as command line parameters\n";
        exit(-1);
    }
    precision = std::string(argv[1]);
    if (argc==3) {
        cnt_rows_A_rows_C = atoi(argv[2]);
        cnt_cols_A_rows_B = cnt_rows_A_rows_C;
        cnt_cols_B_cols_C = cnt_rows_A_rows_C;
    }
    else
    {
        cnt_rows_A_rows_C = atoi(argv[2]);
        cnt_cols_A_rows_B = atoi(argv[3]);
        cnt_cols_B_cols_C = atoi(argv[4]);
    }
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
    fprintf(stderr, "done malloc\n");
    for(i=0; i < cnt_rows_A_rows_C * cnt_cols_A_rows_B; i++) { A[i] = static_cast<float>(rand())/RAND_MAX;}
    for(i=0; i < cnt_cols_A_rows_B * cnt_cols_B_cols_C; i++) { B[i] = static_cast<float>(rand())/RAND_MAX;}
    for(i=0; i < cnt_rows_A_rows_C * cnt_cols_B_cols_C; i++) { C[i] = static_cast<float>(rand())/RAND_MAX;}
    fprintf(stderr, "done random init\n");
}