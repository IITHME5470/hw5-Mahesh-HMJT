#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void grid(int nx, int nxglob, int istglob, int ienglob, double xstglob, double xenglob, double *x, double *dx) {
    int i, iglob;
    *dx = (xenglob - xstglob) / (nxglob - 1);
    for (i = 0; i < nx; i++) {
        iglob = istglob + i;
        x[i] = xstglob + (double)iglob * (*dx);
    }
}

void enforce_bcs(int nx, int ny, double *x, double *y, double **T, int istglob, int ienglob, int jstglob, int jenglob, int nxglob, int nyglob) {
    if (istglob == 0) for (int j = 0; j < ny; j++) T[0][j] = 0.0;
    if (ienglob == nxglob - 1) for (int j = 0; j < ny; j++) T[nx - 1][j] = 0.0;
    if (jstglob == 0) for (int i = 0; i < nx; i++) T[i][0] = 0.0;
    if (jenglob == nyglob - 1) for (int i = 0; i < nx; i++) T[i][ny - 1] = 0.0;
}

void set_initial_condition(int nx, int ny, double *x, double *y, double **T, double dx, double dy) {
    int i, j;
    double del = 1.0;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            T[i][j] = 0.25 * (tanh((x[i] - 0.4) / (del * dx)) - tanh((x[i] - 0.6) / (del * dx))) * 
                      (tanh((y[j] - 0.4) / (del * dy)) - tanh((y[j] - 0.6) / (del * dy)));
}

void get_rhs(int nx, int ny, double dx, double dy, double *xleftghost, double *xrightghost, double *ybotghost, double *ytopghost,
             double kdiff, double **T, double **rhs, int istglob, int ienglob, int jstglob, int jenglob, int nxglob, int nyglob) {
    double dxsq = dx * dx, dysq = dy * dy;
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
            rhs[i][j] = kdiff * ((T[i + 1][j] + T[i - 1][j] - 2 * T[i][j]) / dxsq + 
                                 (T[i][j + 1] + T[i][j - 1] - 2 * T[i][j]) / dysq);
    if (istglob > 0)
        for (int j = 1; j < ny - 1; j++)
            rhs[0][j] = kdiff * ((T[1][j] + xleftghost[j] - 2 * T[0][j]) / dxsq + 
                                 (T[0][j + 1] + T[0][j - 1] - 2 * T[0][j]) / dysq);
    else
        for (int j = 1; j < ny - 1; j++) rhs[0][j] = 0.0;
    if (ienglob < nxglob - 1)
        for (int j = 1; j < ny - 1; j++)
            rhs[nx - 1][j] = kdiff * ((xrightghost[j] + T[nx - 2][j] - 2 * T[nx - 1][j]) / dxsq + 
                                      (T[nx - 1][j + 1] + T[nx - 1][j - 1] - 2 * T[nx - 1][j]) / dysq);
    else
        for (int j = 1; j < ny - 1; j++) rhs[nx - 1][j] = 0.0;
    if (jstglob > 0)
        for (int i = 1; i < nx - 1; i++)
            rhs[i][0] = kdiff * ((T[i + 1][0] + T[i - 1][0] - 2 * T[i][0]) / dxsq + 
                                 (T[i][1] + ybotghost[i] - 2 * T[i][0]) / dysq);
    else
        for (int i = 1; i < nx - 1; i++) rhs[i][0] = 0.0;
    if (jenglob < nyglob - 1)
        for (int i = 1; i < nx - 1; i++)
            rhs[i][ny - 1] = kdiff * ((T[i + 1][ny - 1] + T[i - 1][ny - 1] - 2 * T[i][ny - 1]) / dxsq + 
                                      (ytopghost[i] + T[i][ny - 2] - 2 * T[i][ny - 1]) / dysq);
    else
        for (int i = 1; i < nx - 1; i++) rhs[i][ny - 1] = 0.0;
}

void halo_exchange_2d_x(int rank, int rank_x, int rank_y, int size, int px, int py, int nx, int ny, 
                        double **T, double *xleftghost, double *xrightghost, double *sendbuf_x, double *recvbuf_x) {
    MPI_Status status;
    int left_nb = (rank_x > 0) ? rank - py : MPI_PROC_NULL;
    int right_nb = (rank_x < px - 1) ? rank + py : MPI_PROC_NULL;
    for (int j = 0; j < ny; j++) sendbuf_x[j] = T[1][j];
    MPI_Sendrecv(sendbuf_x, ny, MPI_DOUBLE, left_nb, 0, recvbuf_x, ny, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD, &status);
    for (int j = 0; j < ny; j++) xrightghost[j] = (right_nb == MPI_PROC_NULL) ? 0.0 : recvbuf_x[j];
    for (int j = 0; j < ny; j++) sendbuf_x[j] = T[nx - 2][j];
    MPI_Sendrecv(sendbuf_x, ny, MPI_DOUBLE, right_nb, 1, recvbuf_x, ny, MPI_DOUBLE, left_nb, 1, MPI_COMM_WORLD, &status);
    for (int j = 0; j < ny; j++) xleftghost[j] = (left_nb == MPI_PROC_NULL) ? 0.0 : recvbuf_x[j];
}

void halo_exchange_2d_y(int rank, int rank_x, int rank_y, int size, int px, int py, int nx, int ny, 
                        double **T, double *ybotghost, double *ytopghost, double *sendbuf_y, double *recvbuf_y) {
    MPI_Status status;
    int bot_nb = (rank_y > 0) ? rank - 1 : MPI_PROC_NULL;
    int top_nb = (rank_y < py - 1) ? rank + 1 : MPI_PROC_NULL;
    for (int i = 0; i < nx; i++) sendbuf_y[i] = T[i][1];
    MPI_Sendrecv(sendbuf_y, nx, MPI_DOUBLE, bot_nb, 2, recvbuf_y, nx, MPI_DOUBLE, top_nb, 2, MPI_COMM_WORLD, &status);
    for (int i = 0; i < nx; i++) ytopghost[i] = (top_nb == MPI_PROC_NULL) ? 0.0 : recvbuf_y[i];
    for (int i = 0; i < nx; i++) sendbuf_y[i] = T[i][ny - 2];
    MPI_Sendrecv(sendbuf_y, nx, MPI_DOUBLE, top_nb, 3, recvbuf_y, nx, MPI_DOUBLE, bot_nb, 3, MPI_COMM_WORLD, &status);
    for (int i = 0; i < nx; i++) ybotghost[i] = (bot_nb == MPI_PROC_NULL) ? 0.0 : recvbuf_y[i];
}

void timestep_FwdEuler(int rank, int size, int rank_x, int rank_y, int px, int py, int nx, int nxglob, int ny, int nyglob,
                       int istglob, int ienglob, int jstglob, int jenglob, double dt, double dx, double dy,
                       double *xleftghost, double *xrightghost, double *ybotghost, double *ytopghost, double kdiff,
                       double *x, double *y, double **T, double **rhs, double *sendbuf_x, double *recvbuf_x,
                       double *sendbuf_y, double *recvbuf_y) {
    halo_exchange_2d_x(rank, rank_x, rank_y, size, px, py, nx, ny, T, xleftghost, xrightghost, sendbuf_x, recvbuf_x);
    halo_exchange_2d_y(rank, rank_x, rank_y, size, px, py, nx, ny, T, ybotghost, ytopghost, sendbuf_y, recvbuf_y);
    get_rhs(nx, ny, dx, dy, xleftghost, xrightghost, ybotghost, ytopghost, kdiff, T, rhs, 
            istglob, ienglob, jstglob, jenglob, nxglob, nyglob);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            T[i][j] += dt * rhs[i][j];
    enforce_bcs(nx, ny, x, y, T, istglob, ienglob, jstglob, jenglob, nxglob, nyglob);
}

void output_soln(int rank, int nx, int ny, int it, double tcurr, double *x, double *y, double **T, const char *folder) {
    char fname[100];
    sprintf(fname, "%s/T_x_y_%06d_%02d.dat", folder, it, rank);
    FILE *fp = fopen(fname, "w");
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            fprintf(fp, "%lf %lf %lf\n", x[i], y[j], T[i][j]);
    fclose(fp);
}

void get_processor_grid_ranks(int rank, int size, int px, int py, int *rank_x, int *rank_y) {
    *rank_y = rank % py;
    *rank_x = rank / py;
}

int main(int argc, char **argv) {
    int nx, ny, nxglob, nyglob, rank, size, px, py, rank_x, rank_y;
    double *x, *y, **T, **rhs, tst, ten, xstglob, xenglob, ystglob, yenglob, dx, dy, dt, tcurr, kdiff;
    double xst, yen, xlen, ylen, min_dx_dy;
    double *xleftghost, *xrightghost, *ybotghost, *ytopghost;
    double *sendbuf_x, *sendbuf_y, *recvbuf_x, *recvbuf_y;
    int i, it, num_time_steps, istglob, ienglob, jstglob, jenglob;
    char folder[10];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        FILE *fid = fopen("input2d.in", "r");
        if (!fid) { printf("ERROR: Could not open input2d.in\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fscanf(fid, "%d %d\n", &nxglob, &nyglob);
        fscanf(fid, "%lf %lf %lf %lf\n", &xstglob, &xenglob, &ystglob, &yenglob);
        fscanf(fid, "%lf %lf %lf %lf\n", &tst, &ten, &dt, &kdiff); // t_print not used
        fscanf(fid, "%lf\n", &kdiff);
        fscanf(fid, "%d %d\n", &px, &py);
        fclose(fid);

        if (nxglob % px != 0 || nyglob % py != 0) {
            printf("ERROR: Grid dimensions must be divisible by processor grid!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double dx_global = (xenglob - xstglob) / (nxglob - 1);
        double dy_global = (yenglob - ystglob) / (nyglob - 1);
        min_dx_dy = fmin(dx_global, dy_global);
        dt = 0.25 * min_dx_dy * min_dx_dy / (2.0 * kdiff);
        num_time_steps = 5107; // Match your previous runs
    }

    int params_int[8] = {nxglob, nyglob, px, py, num_time_steps, 0, 0, 0};
    MPI_Bcast(params_int, 8, MPI_INT, 0, MPI_COMM_WORLD);
    nxglob = params_int[0];
    nyglob = params_int[1];
    px = params_int[2];
    py = params_int[3];
    num_time_steps = params_int[4];

    double params_dbl[9] = {tst, ten, dt, xstglob, xenglob, ystglob, yenglob, kdiff, 0};
    MPI_Bcast(params_dbl, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    tst = params_dbl[0];
    ten = params_dbl[1];
    dt = params_dbl[2];
    xstglob = params_dbl[3];
    xenglob = params_dbl[4];
    ystglob = params_dbl[5];
    yenglob = params_dbl[6];
    kdiff = params_dbl[7];

    get_processor_grid_ranks(rank, size, px, py, &rank_x, &rank_y);
    nx = nxglob / px;
    ny = nyglob / py;
    xlen = (xenglob - xstglob) / px;
    ylen = (yenglob - ystglob) / py;

    istglob = rank_x * nx;
    ienglob = (rank_x + 1) * nx - 1;
    jstglob = rank_y * ny;
    jenglob = (rank_y + 1) * ny - 1;
    xst = xstglob + rank_x * xlen;
    yen = ystglob + rank_y * ylen + ylen;

    x = malloc(nx * sizeof(double));
    y = malloc(ny * sizeof(double));
    T = malloc(nx * sizeof(double *));
    rhs = malloc(nx * sizeof(double *));
    for (i = 0; i < nx; i++) {
        T[i] = malloc(ny * sizeof(double));
        rhs[i] = malloc(ny * sizeof(double));
    }
    xleftghost = malloc(ny * sizeof(double));
    xrightghost = malloc(ny * sizeof(double));
    ybotghost = malloc(nx * sizeof(double));
    ytopghost = malloc(nx * sizeof(double));
    sendbuf_x = malloc(ny * sizeof(double));
    recvbuf_x = malloc(ny * sizeof(double));
    sendbuf_y = malloc(nx * sizeof(double));
    recvbuf_y = malloc(nx * sizeof(double));

    for (i = 0; i < ny; i++) { xleftghost[i] = xrightghost[i] = 0.0; }
    for (i = 0; i < nx; i++) { ybotghost[i] = ytopghost[i] = 0.0; }

    grid(nx, nxglob, istglob, ienglob, xstglob, xenglob, x, &dx);
    grid(ny, nyglob, jstglob, jenglob, ystglob, yenglob, y, &dy);
    set_initial_condition(nx, ny, x, y, T, dx, dy);
    enforce_bcs(nx, ny, x, y, T, istglob, ienglob, jstglob, jenglob, nxglob, nyglob);

    sprintf(folder, "%d_%d", px, py);
    output_soln(rank, nx, ny, 0, tst, x, y, T, folder);

    double start, end, total_time = 0.0;
    for (it = 0; it < num_time_steps; it++) {
        start = MPI_Wtime();
        tcurr = tst + (it + 1) * dt;
        timestep_FwdEuler(rank, size, rank_x, rank_y, px, py, nx, nxglob, ny, nyglob,
                          istglob, ienglob, jstglob, jenglob, dt, dx, dy,
                          xleftghost, xrightghost, ybotghost, ytopghost, kdiff,
                          x, y, T, rhs, sendbuf_x, recvbuf_x, sendbuf_y, recvbuf_y);
        end = MPI_Wtime();
        total_time += (end - start);
        if (it == 509 || it == 1019 || it == 1529 || it == 2039 || it == 2549 || 
            it == 3059 || it == 3569 || it == 4079 || it == 4589 || it == 5106) {
            output_soln(rank, nx, ny, it + 1, tcurr, x, y, T, folder);
        }
    }

    double avg_time_per_step = total_time / num_time_steps;
    if (rank == 0) {
        char timing_file[100];
        sprintf(timing_file, "%s/timing.txt", folder);
        FILE *fp = fopen(timing_file, "w");
        fprintf(fp, "%dx%d: Average time per time step = %e seconds\n", px, py, avg_time_per_step);
        fclose(fp);
        printf("%dx%d: Average time per time step = %e seconds\n", px, py, avg_time_per_step);
    }

    free(x); free(y); free(xleftghost); free(xrightghost); free(ybotghost); free(ytopghost);
    free(sendbuf_x); free(recvbuf_x); free(sendbuf_y); free(recvbuf_y);
    for (i = 0; i < nx; i++) { free(T[i]); free(rhs[i]); }
    free(T); free(rhs);

    MPI_Finalize();
    return 0;
}