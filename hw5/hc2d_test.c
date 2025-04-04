#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void grid(int nx, double xst, double xen, double *x, double *dx)
{
  int i;
  
  *dx = (xen-xst)/(double)(nx-1);

  for(i=0; i<nx; i++)
    x[i] = xst + (double)i * (*dx); // Modified to properly account for xst
}

void enforce_bcs(int nx, int ny, double *x, double *y, double **T)
{
  int i, j;

  // left and right ends
  for(j=0; j<ny; j++)
  {
    T[0][j] = 0.0;    T[nx-1][j] = 0.0;
  }

  // top and bottom ends
  for(i=0; i<nx; i++)
  {
    T[i][0] = 0.0;    T[i][ny-1] = 0.0;
  }
}

void set_initial_condition(int nx, int ny, double *x, double *y, double **T, double dx, double dy)
{
  int i, j;
  double del=1.0;

  for(i=0; i<nx; i++)
    for(j=0; j<ny; j++)
    {
        T[i][j] = 0.25 * (tanh((x[i]-0.4)/(del*dx)) - tanh((x[i]-0.6)/(del*dx))) 
                       * (tanh((y[j]-0.4)/(del*dy)) - tanh((y[j]-0.6)/(del*dy)));
    }

  // Debug print for initial condition
  printf("Initial condition set. T[%d][%d] = %f\n", nx/2, ny/2, T[nx/2][ny/2]);
  
  enforce_bcs(nx,ny,x,y,T);
}

void timestep_FwdEuler(int nx, int ny, double dt, double dx, double dy, double kdiff, double *x, double *y, double **T, double **rhs)
{
  int i,j;
  double dxsq = dx*dx, dysq = dy*dy;

  for(i=1; i<nx-1; i++)
   for(j=1; j<ny-1; j++)
     rhs[i][j] = kdiff*(T[i+1][j]+T[i-1][j]-2.0*T[i][j])/dxsq +
           kdiff*(T[i][j+1]+T[i][j-1]-2.0*T[i][j])/dysq;

  for(i=1; i<nx-1; i++)
   for(j=1; j<ny-1; j++)
     T[i][j] = T[i][j] + dt*rhs[i][j];

  enforce_bcs(nx,ny,x,y,T);
}

double get_error_norm_2d(int nx, int ny, double **arr1, double **arr2)
{
  double norm_diff = 0.0, local_diff;
  int i, j;

  for(i=0; i<nx; i++)
   for(j=0; j<ny; j++)
   {
     local_diff = arr1[i][j] - arr2[i][j];
     norm_diff += local_diff * local_diff;
   }
   norm_diff = sqrt(norm_diff/(double) (nx*ny));
   return norm_diff; // Added missing return statement
}

void linsolve_hc2d_gs(int nx, int ny, double rx, double ry, double **rhs, double **T, double **Tnew)
{
  int i, j, k, max_iter;
  double tol, denom, norm_diff;

  max_iter = 1000; tol = 1.0e-6;
  denom = 1.0 + 2.0*rx + 2.0*ry;

  for(k=0; k<max_iter;k++)
  {
    for(i=1; i<nx-1; i++)
     for(j=1; j<ny-1; j++)
       Tnew[i][j] = (rhs[i][j] + rx*Tnew[i-1][j] + rx*T[i+1][j] + ry*Tnew[i][j-1] + ry*T[i][j+1]) /denom;

    norm_diff = get_error_norm_2d(nx, ny, T, Tnew);
    if(norm_diff < tol) break;

    for(i=0; i<nx; i++)
     for(j=0; j<ny; j++)
       T[i][j] = Tnew[i][j];
  }
  printf("GS solver: iterations=%d, residual=%e\n", k, norm_diff);
}

void output_soln(int nx, int ny, int it, double tcurr, double *x, double *y, double **T)
{
  int i,j;
  FILE* fp;
  char fname[100];

  sprintf(fname, "T_x_y_%06d.dat", it);
  printf("Attempting to write output file: %s\n", fname); // Debug print

  fp = fopen(fname, "w");
  if(fp == NULL) {
    perror("Error opening output file");
    return;
  }
  
  for(i=0; i<nx; i++)
   for(j=0; j<ny; j++)
      fprintf(fp, "%lf %lf %lf\n", x[i], y[j], T[i][j]);
  
  fclose(fp);
  printf("Successfully wrote solution for timestep=%d, t=%e\n", it, tcurr);
}

int main()
{
  int nx, ny;
  double *x, *y, **T, **rhs, tst, ten, xst, xen, yst, yen, dx, dy, dt, tcurr, kdiff;
  double min_dx_dy, **Tnew;
  int i, it, num_time_steps, it_print, j;
  FILE* fp;  

  // Read inputs
  fp = fopen("input2d_ser.in", "r");
  if(fp == NULL) {
    perror("Error opening input file");
    return 1;
  }
  fscanf(fp, "%d %d\n", &nx, &ny);
  fscanf(fp, "%lf %lf %lf %lf\n", &xst, &xen, &yst, &yen);
  fscanf(fp, "%lf %lf\n", &tst, &ten);
  fscanf(fp, "%lf\n", &kdiff);
  fclose(fp);

  printf("Input parameters:\n");
  printf("nx=%d, ny=%d\n", nx, ny);
  printf("x range: [%lf, %lf]\n", xst, xen);
  printf("y range: [%lf, %lf]\n", yst, yen);
  printf("t range: [%lf, %lf]\n", tst, ten);
  printf("kdiff=%lf\n", kdiff);

  // Allocate memory
  x = (double *)malloc(nx*sizeof(double));
  y = (double *)malloc(ny*sizeof(double));
  T = (double **)malloc(nx*sizeof(double *));
  rhs = (double **)malloc(nx*sizeof(double *));
  Tnew = (double **)malloc(nx*sizeof(double *));
  for(i=0; i<nx; i++) {
    T[i] = (double *)malloc(ny*sizeof(double));
    rhs[i] = (double *)malloc(ny*sizeof(double));
    Tnew[i] = (double *)malloc(ny*sizeof(double));
  }

  // Initialize grid and solution
  grid(nx,xst,xen,x,&dx);
  grid(ny,yst,yen,y,&dy);
  set_initial_condition(nx,ny,x,y,T,dx,dy);

  // Time stepping setup
  min_dx_dy = fmin(dx, dy);
  dt = 0.25 * min_dx_dy * min_dx_dy /(2 * kdiff) ; // More stable time step
  num_time_steps = (int)((ten-tst)/dt) + 1;
  // it_print = num_time_steps > 5 ? num_time_steps/5 : 1; // Ensure at least 1 output
  it_print = (int)(0.0004 /(4*dt));                     // ≈ 2040
  if (it_print == 0) it_print = 1;

  printf("Time stepping parameters:\n");
  printf("dx=%lf, dy=%lf\n", dx, dy);
  printf("dt=%lf\n", dt);
  printf("num_time_steps=%d\n", num_time_steps);
  printf("it_print=%d\n", it_print);

  // Time stepping loop
  for(it=0; it<num_time_steps; it++)
  {
    tcurr = tst + (double)(it+1) * dt;
    // printf("Timestep %d/%d, t=%lf\n", it+1, num_time_steps, tcurr); // Progress indicator

    timestep_FwdEuler(nx,ny,dt,dx,dy,kdiff,x,y,T,rhs);

    if(it%it_print==0 || it==num_time_steps-1) {
      output_soln(nx,ny,it,tcurr,x,y,T);
    }
  }

  // Free memory
  for(i=0; i<nx; i++) {
    free(T[i]);
    free(rhs[i]);
    free(Tnew[i]);
  }
  free(T);
  free(rhs);
  free(Tnew);
  free(y);
  free(x);

  return 0;
}