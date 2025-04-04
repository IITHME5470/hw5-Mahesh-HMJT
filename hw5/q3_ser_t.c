#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void grid(int nx, double xst, double xen, double *x, double *dx)
{
    int i;
    *dx = (xen - xst) / (double)(nx - 1);
    for (i = 0; i < nx; i++)
        x[i] = xst + (double)i * (*dx);
}

void enforce_bcs(int nx, int ny, double *x, double *y, double **T)
{
    int i, j;
    for (j = 0; j < ny; j++) {
        T[0][j] = 0.0;
        T[nx - 1][j] = 0.0;
    }
    for (i = 0; i < nx; i++) {
        T[i][0] = 0.0;
        T[i][ny - 1] = 0.0;
    }
}

void set_initial_condition(int nx, int ny, double *x, double *y, double **T, double dx, double dy)
{
    int i, j;
    double del = 1.0;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            T[i][j] = 0.25 * (tanh((x[i] - 0.4) / (del * dx)) - tanh((x[i] - 0.6) / (del * dx))) *
                      (tanh((y[j] - 0.4) / (del * dy)) - tanh((y[j] - 0.6) / (del * dy)));
    enforce_bcs(nx, ny, x, y, T);
}

void timestep_FwdEuler(int nx, int ny, double dt, double dx, double dy, double kdiff, double *x, double *y, double **T, double **rhs)
{
    int i, j;
    double dxsq = dx * dx, dysq = dy * dy;
    for (i = 1; i < nx - 1; i++)
        for (j = 1; j < ny - 1; j++)
            rhs[i][j] = kdiff * (T[i + 1][j] + T[i - 1][j] - 2.0 * T[i][j]) / dxsq +
                        kdiff * (T[i][j + 1] + T[i][j - 1] - 2.0 * T[i][j]) / dysq;
    for (i = 1; i < nx - 1; i++)
        for (j = 1; j < ny - 1; j++)
            T[i][j] = T[i][j] + dt * rhs[i][j];
    enforce_bcs(nx, ny, x, y, T);
}

void output_soln(int nx, int ny, int it, double tcurr, double *x, double *y, double **T)
{
    int i, j;
    FILE *fp;
    char fname[100];
    sprintf(fname, "ser/T_x_y_%06d.dat", it);
    fp = fopen(fname, "w");
    if (fp == NULL) {
        perror("Error opening output file");
        return;
    }
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            fprintf(fp, "%lf %lf %lf\n", x[i], y[j], T[i][j]);
    fclose(fp);
}

int main()
{
    int nx, ny;
    double *x, *y, **T, **rhs, tst, ten, xst, xen, yst, yen, dx, dy, dt, tcurr, kdiff;
    double min_dx_dy;
    int i, it, num_time_steps, it_print;
    FILE *fp;
    clock_t start, end;
    double total_time = 0.0;

    // Read inputs
    fp = fopen("input2d_ser.in", "r");
    if (fp == NULL) {
        perror("Error opening input file");
        return 1;
    }
    fscanf(fp, "%d %d\n", &nx, &ny);
    fscanf(fp, "%lf %lf %lf %lf\n", &xst, &xen, &yst, &yen);
    fscanf(fp, "%lf %lf\n", &tst, &ten);
    fscanf(fp, "%lf\n", &kdiff);
    fclose(fp);

    // Allocate memory
    x = (double *)malloc(nx * sizeof(double));
    y = (double *)malloc(ny * sizeof(double));
    T = (double **)malloc(nx * sizeof(double *));
    rhs = (double **)malloc(nx * sizeof(double *));
    for (i = 0; i < nx; i++) {
        T[i] = (double *)malloc(ny * sizeof(double));
        rhs[i] = (double *)malloc(ny * sizeof(double));
    }

    // Initialize grid and solution
    grid(nx, xst, xen, x, &dx);
    grid(ny, yst, yen, y, &dy);
    set_initial_condition(nx, ny, x, y, T, dx, dy);
    output_soln(nx, ny, 0, tst, x, y, T);

    // Time stepping setup
    min_dx_dy = fmin(dx, dy);
    dt = 0.25 * min_dx_dy * min_dx_dy / (2 * kdiff);
    num_time_steps = (int)((ten - tst) / dt) + 1;
    it_print = (int)(0.0004 / (4 * dt)); // Matches your previous 2040
    if (it_print == 0) it_print = 1;

    // Time stepping loop with timing
    for (it = 0; it < num_time_steps; it++) {
        start = clock();
        tcurr = tst + (double)(it + 1) * dt;
        timestep_FwdEuler(nx, ny, dt, dx, dy, kdiff, x, y, T, rhs);
        end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC;

        // Output at specified time steps
        if (it == 509 || it == 1019 || it == 1529 || it == 2039 || it == 2549 ||
            it == 3059 || it == 3569 || it == 4079 || it == 4589 || it == 5106) {
            output_soln(nx, ny, it + 1, tcurr, x, y, T);
        }
    }

    // Compute and output average time per step
    double avg_time_per_step = total_time / num_time_steps;
    fp = fopen("ser/timing.txt", "w");
    if (fp == NULL) {
        perror("Error opening timing file");
        return 1;
    }
    fprintf(fp, "Serial: Average time per time step = %e seconds\n", avg_time_per_step);
    fclose(fp);
    printf("Serial: Average time per time step = %e seconds\n", avg_time_per_step);

    // Free memory
    for (i = 0; i < nx; i++) {
        free(T[i]);
        free(rhs[i]);
    }
    free(T);
    free(rhs);
    free(x);
    free(y);

    return 0;
}