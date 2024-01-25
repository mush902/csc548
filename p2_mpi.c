//Group info:
//mshaikh2 Mushtaq Ahmed Shaikh
//ahgada Amay Gada

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

/* first grid point */
#define   XI              1.0
/* last grid point */
#define   XF              100.0


double fn(double);
void write_to_file(int , FILE *, double *, double * , double *);
void write_to_file_mpigather(int , FILE *, double *, double * , double *);

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator (MPI_COMM_WORLD)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Input parameters
    int NGRID;
    int comm_type, gather_type;
    
    // Check if command-line arguments are provided
    if (argc > 2) {
        NGRID = atoi(argv[1]);
        comm_type = atoi(argv[2]);
        gather_type = atoi(argv[3]);
    }else{
        printf("Please specify the number of GRID POINTS, P2P COMMUNICATION TYPE, and GATHER TYPE.\n");
    }

    // Calculate local grid size for each process
    double step_size = (XF - XI) / (NGRID - 1);

    double grid_points[NGRID];

    for (int i = 0; i < NGRID; i++) {
        grid_points[i] = XI + (i*step_size);
    }

    int base_size =NGRID / size;
    int remainder = NGRID % size;

    int local_NGRID;
    if(rank ==size-1) {
        local_NGRID = base_size + remainder;
    }
    else {
        local_NGRID = base_size;
    }


    // Calculate local indices
    int local_imin = 1;
    int local_imax = local_NGRID;

    // Allocate memory for local arrays
    double* xc = (double*)malloc((local_NGRID + 2) * sizeof(double));
    double* yc = (double*)malloc((local_NGRID + 2) * sizeof(double));
    double* dyc = (double*)malloc((local_NGRID) * sizeof(double));
    double dx;

    int start_index = rank*base_size;

    for (int i = 0; i < local_NGRID; i++) {
        xc[i+1] = grid_points[start_index + i];
    }

    for (int i = 1; i <= local_NGRID; i++) {
        yc[i] = fn(xc[i]);
    }

    dx = xc[local_imin+1]-xc[local_imin]; //this is step size 

    xc[local_imax+1] = xc[local_imax]+dx;

    if (rank == 0) {
        //dx = xc[2] - xc[1];
        xc[local_imin-1] = xc[1] - dx;
        yc[local_imin-1] = fn(xc[local_imin-1]);
    }

    if (rank == size - 1) {
        //dx = xc[2] - xc[1];
        xc[local_imax + 1] = xc[local_imax] + dx;
        yc[local_imax + 1] = fn(xc[local_imax + 1]);
    }

    // Communication: Exchange boundary values of yc with neighbors
    if(comm_type == 0) {
        if (rank > 0) {
            // Send yc[local_imin] to the left neighbor (rank - 1)
            MPI_Send(&yc[local_imin], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);

            // Receive yc[local_imin - 1] from the left neighbor
            MPI_Recv(&yc[local_imin - 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank < size - 1) {
            // Receive yc[local_imax + 1] from the right neighbor
            MPI_Recv(&yc[local_imax + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send yc[local_imax] to the right neighbor (rank + 1)
            MPI_Send(&yc[local_imax], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        }
    }
    else if (comm_type == 1) {
        // Communication: Exchange boundary values of yc with neighbors
        MPI_Request send_request, recv_request;

        if (rank > 0) {
            // Initiate non-blocking send of yc[local_imin] to the left neighbor (rank - 1)
            MPI_Isend(&yc[local_imin], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_request);

            // Initiate non-blocking receive of yc[local_imin - 1] from the left neighbor
            MPI_Irecv(&yc[local_imin - 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_request);
        }

        if (rank < size - 1) {
            // Initiate non-blocking receive of yc[local_imax + 1] from the right neighbor
            MPI_Irecv(&yc[local_imax + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_request);

            // Initiate non-blocking send of yc[local_imax] to the right neighbor (rank + 1)
            MPI_Isend(&yc[local_imax], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_request);
        }

        // Wait for the non-blocking communication to complete
        if (rank > 0) {
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        }

        if (rank < size - 1) {
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        }
    }else{
        printf("Invalid comm type\n");
        exit(0);
    }


    // Adjust indices for xc, yc, and dyc based on the exchanged boundary values
    int imin = 1;
    int imax = local_NGRID;

    // Compute derivatives using finite differencing
    for (int i = imin; i <= imax; i++) {
        dyc[i-1] = (yc[i + 1] - yc[i - 1]) / (2.0 * (xc[i + 1] - xc[i]));
    }

    // Send all data to rank 0 via gather
    if(gather_type == 0){
        //MPI_GATHER

        // clipping xc, and yc arrays (personal implementation overhead)
        double xc_new[base_size];
        double yc_new[base_size];
        double dy_new[base_size];
        for(int ii=0; ii<base_size; ii++){
            xc_new[ii] = xc[ii+1];
            yc_new[ii] = yc[ii+1];
            dy_new[ii] = dyc[ii];
        }

        
        // define array pointers to store aggregated results at rank 0
        double *x_final = NULL;
        double *y_final = NULL;
        double *dy_final = NULL;
        
        if (rank == 0) {
            x_final = malloc(sizeof(double) * NGRID);
            y_final = malloc(sizeof(double) * NGRID);
            dy_final = malloc(sizeof(double) * NGRID);
        }

        MPI_Gather(xc_new, base_size, MPI_DOUBLE, x_final, base_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(yc_new, base_size, MPI_DOUBLE, y_final, base_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(dy_new, base_size, MPI_DOUBLE, dy_final, base_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //handle case where extra data needs to be pulled from rank = size-1 -> due to processor not evenly dividing the Grids
        if(remainder!=0 && rank==size-1){
            double remaining_x[remainder];
            double remaining_y[remainder];
            double remaining_dy[remainder];

            int k=0;

            for (int i = base_size; i < local_NGRID; i++) {
                remaining_x[k] = xc[1+i];
                remaining_y[k] = yc[1+i];
                remaining_dy[k] = dyc[i]; 
                k+=1;
            }

            MPI_Send(remaining_x, remainder, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(remaining_y, remainder, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(remaining_dy, remainder, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }

        if(rank == 0){
            //define the file name to write data to. We will write data as rank 0 receives it. Gathering here is quite serial/
            char filename[1024];
            sprintf(filename, "fn-%d.dat", NGRID);
            FILE *fp = fopen(filename, "w");

            write_to_file_mpigather(base_size*size, fp, x_final, y_final, dy_final);

            if(remainder != 0){
                // allocating space for incoming data
                double temp_x[remainder];
                double temp_y[remainder];
                double temp_dy[remainder];

                // receiving data
                MPI_Recv(temp_x, remainder, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(temp_y, remainder, MPI_DOUBLE, size-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(temp_dy, remainder, MPI_DOUBLE, size-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                write_to_file_mpigather(remainder, fp, temp_x, temp_y, temp_dy);
            }

            fclose(fp);
        }



    }else if(gather_type == 1){
        //Manual Gather

        if(rank != 0){
            //send to rank 0
            MPI_Send(xc, local_NGRID+2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(yc, local_NGRID+2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(dyc, local_NGRID, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            
        }else{
            //define the file name to write data to. We will write data as rank 0 receives it. Gathering here is quite serial/
            char filename[1024];
            sprintf(filename, "fn-%d.dat", NGRID);
            FILE *fp = fopen(filename, "w");

            // Since 0 is the first in the order, write it to fn-NGRID.dat
            write_to_file(local_NGRID, fp, xc, yc, dyc);

            // loop through all ranks excet 0 to receive data from them
            for(int r=1; r<size; r++){
                int recv_amount = base_size; // number of values to be received
                if(r == size-1) recv_amount += remainder;

                // allocating space for incoming data
                double temp_x[recv_amount+2];
                double temp_y[recv_amount+2];
                double temp_dy[recv_amount];

                // receiving data
                MPI_Recv(temp_x, recv_amount+2, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(temp_y, recv_amount+2, MPI_DOUBLE, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(temp_dy, recv_amount, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // writing data directly to file without extra overhead of saving entire array first
                write_to_file(recv_amount, fp, temp_x, temp_y, temp_dy);
            }

            fclose(fp);
        }


    }else{
        printf("Invalid Gather type\n");
        exit(0);
    }


    MPI_Finalize();

}

// writes to the file, given pointer
void write_to_file(int np, FILE *fp, double *x, double *y, double *dydx)
{       
        int   i;

        for(i = 0; i < np; i++)
        {
                fprintf(fp, "%f %f %f\n", x[i+1], y[i+1], dydx[i]);
        }
}


// slightly different write to file due to implementation overhead
void write_to_file_mpigather(int np, FILE *fp, double *x, double *y, double *dydx)
{       
        int   i;

        for(i = 0; i < np; i++)
        {
                fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
        }
}
