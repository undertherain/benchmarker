    size_t m, n, k;
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
            return -1;
    t_float *A, *B, *C;
    double dtime;
    size_t i;
    A = (t_float*) malloc(sizeof(t_float) * m * n);
    B = (t_float*) malloc(sizeof(t_float) * n * k);
    C = (t_float*) malloc(sizeof(t_float) * m * k);
    for(i=0; i < m * n; i++) { A[i] = rand()/RAND_MAX;}
    for(i=0; i < n * k; i++) { B[i] = rand()/RAND_MAX;}
    for(i=0; i < m * k; i++) { C[i] = rand()/RAND_MAX;}
    fprintf(stderr, "done random init\n");
