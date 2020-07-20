// traditional BLAS API:
// M - Number of rows in matrices A and C.
// N - Number of columns in matrices B and C.
// K - Number of columns in matrix A; number of rows in matrix B.

void parse_args(const int argc,
                char *argv[],
                std::string &precision,
                size_t &m,
                size_t &n,
                size_t &k
                ) {
    if ((argc != 3) && (argc != 5))
    {
        std::cerr << "provide precision, m, n, k as command line parameters\n";
        throw "provide precision, m, n, k as command line parameters";
    }
    precision = std::string(argv[1]);
    if (argc==3) {
        m = atoi(argv[2]);
        n = m;
        k = m;
    }
    else
    {
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }
}

template<typename precision>
void get_matrices(size_t &m,
                  size_t &n,
                  size_t &k,
                  precision * & A,
                  precision * & B,
                  precision * & C) {
    size_t i;
    A = (precision*) malloc(sizeof(precision) * m * n);
    B = (precision*) malloc(sizeof(precision) * n * k);
    C = (precision*) malloc(sizeof(precision) * m * k);
    for(i=0; i < m * n; i++) { A[i] = rand()/RAND_MAX;}
    for(i=0; i < n * k; i++) { B[i] = rand()/RAND_MAX;}
    for(i=0; i < m * k; i++) { C[i] = rand()/RAND_MAX;}
    fprintf(stderr, "done random init\n");
}