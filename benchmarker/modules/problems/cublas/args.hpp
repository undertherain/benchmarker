void args_to_matrices(int argc, char *argv[], size_t &m, size_t &n, size_t &k, float * & A, float *& B, float * & C) {
    if (argc==2) {
        m = atoi(argv[1]);
        n = m;
        k = m;
    }
    else
        if (argc==4) {
            m = atoi(argv[1]);
            n = atoi(argv[2]);
            k = atoi(argv[3]);
        }
        else
            std:cerr << "provide m, n, k as command line parameters\n";
            throw "provide m, n, k as command line parameters";
    size_t i;
    A = (t_float*) malloc(sizeof(t_float) * m * n);
    B = (t_float*) malloc(sizeof(t_float) * n * k);
    C = (t_float*) malloc(sizeof(t_float) * m * k);
    for(i=0; i < m * n; i++) { A[i] = rand()/RAND_MAX;}
    for(i=0; i < n * k; i++) { B[i] = rand()/RAND_MAX;}
    for(i=0; i < m * k; i++) { C[i] = rand()/RAND_MAX;}
    fprintf(stderr, "done random init\n");
}