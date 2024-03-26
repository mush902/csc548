#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "jemalloc/jemalloc.h"

#define BACK_FILE "/tmp/mshaikh2.app.back"
#define MMAP_FILE "/tmp/mshaikh2.app.mmap"
#define MMAP_SIZE ((size_t)1 << 30)

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

int npoints = 128;
int npebs = 8;
double end_time = 1;
int nthreads = 4;
int narea = 128*128;
PERM double u_cpu[128*128], u_i0[128*128], u_i1[128*128], pebs[128*128], t;
PERM int iteration;

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);

int main(int argc, char *argv[])
{

  //Hard coded these variables to global variables as required
  // int     npoints   = atoi(argv[1]);
  // int     npebs     = atoi(argv[2]);
  // double  end_time  = (double)atof(argv[3]);
  // int     nthreads  = atoi(argv[4]);
  // int 	  narea	    = npoints * npoints;

  //Moved their declartion to global scope and used Perm to allocate memory
  // double *u_i0, *u_i1;
  // double *u_cpu, *u_gpu, *pebs;
  //PERM memory already intialised in global space
  // u_i0 = (double*)malloc(sizeof(double) * narea);
  // u_i1 = (double*)malloc(sizeof(double) * narea);
  // pebs = (double*)malloc(sizeof(double) * narea);

  // u_cpu = (double*)malloc(sizeof(double) * narea);
  // u_gpu = (double*)malloc(sizeof(double) * narea);


  int do_restore = argc > 1 && strcmp("-r", argv[1]) == 0;
  const char *mode = (do_restore) ? "r+" : "w+"; 	
  perm(PERM_START, PERM_SIZE);
  mopen(MMAP_FILE, mode, MMAP_SIZE);
  bopen(BACK_FILE, mode);
  printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);
  double h = (XMAX - XMIN)/npoints;
  
  if (!do_restore) {
    iteration = 0;
    t = 0.; 
    init_pebbles(pebs, npebs, npoints);
    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);
    print_heatmap("lake_i.dat", u_i0, npoints, h);
    mflush();
    backup();
  }
  else {
    printf("restarting...\n");
    restore();
  }
  //Move this code inside the if condition so that it executes once during first run and does not execute on subsequent restarts. 
  // init_pebbles(pebs, npebs, npoints);
  // init(u_i0, pebs, npoints);
  // init(u_i1, pebs, npoints);
  // print_heatmap("lake_i.dat", u_i0, npoints, h);

  
  run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);

  print_heatmap("lake_f.dat", u_cpu, npoints, h);

  // Cleanup
  mclose();
  bclose();
  remove(BACK_FILE);
  remove(MMAP_FILE);

  return 1;
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  double *un, *uc, *uo, dt;

  un = (double*)malloc(sizeof(double) * n * n);
  
  dt = h / 2.;
  uo = u0;
  uc = u1;
  while(1)
  {
    printf("Iteration: %d\n",iteration++);
    evolve9pt(un, uc, uo, pebbles, n, h, dt, t);

    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
    
    backup();
  }
  
  memcpy(u, un, sizeof(double) * n * n);
}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand(1024 );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }
}

double f(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

void init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + 
                    uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}

void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1 || i==1 || i==n-2 || j==1 || j==n-2)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *
        (( uc[idx-1] + uc[idx+1] + uc[idx-n] + uc[idx+n] +
        0.25*(uc[idx-n-1] + uc[idx-n+1] + uc[idx+n-1] + uc[idx+n+1]) +
        0.125*(uc[idx-1-1] + uc[idx+1+1] + uc[idx-n-n] + uc[idx+n+n]) -
        5.5 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}


void print_heatmap(const char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");  

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }
  
  fclose(fp);
} 

