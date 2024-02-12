// Single Author info:
// mshaikh2 Mushtaq Ahmed Shaikh


#include <stdlib.h>
#include <stdio.h>
#include "my_mpi.h"
#include <sys/stat.h>

#define MESSAGE_SIZES 7
#define ITERATIONS 100
#define DIRECTORY_NAME "csv_plot_directory"

void createDirectory(const char* dir_name) {
    // Create the output directory if it doesn't exist
    // The output directory contains all the files with their rtt times
    struct stat st = {0};
    if (stat(dir_name, &st) == -1) {
        mkdir(dir_name, 0700);
    }
}

void writeDataToCSV2(double timings[], int msg_size, int iterations, int rank, int size, const char* base_filename) {
    char filename[100];
    // The format of file name is "output_<inter/intera>_size_<message_size>_rank_<rank_number>"
    // for instance file is "output_inter_32768_size_2_rank_0.txt" which means it is inter comm for 32KB message size for scenario 1 of 2 cores and rank is 0
    snprintf(filename, sizeof(filename), "%s/%s_%d_size_%d_rank_%d.csv", DIRECTORY_NAME, base_filename, msg_size, size, rank);

    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return;
    }

    // Write data
    for (int j = 0; j < iterations - 1; j++) {
        fprintf(fp, "%lf\n", timings[j]);
    }

    fclose(fp);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    createDirectory(DIRECTORY_NAME);

    // Check if the total number of processes is 2
    if (size == 2) {
        //printf("There are 2 processes on 2 nodes \n");
	//message size goes from 32KB till 2 MB
        int msg_sizes[MESSAGE_SIZES] = {
            32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
            1 * 1024 * 1024, 2 * 1024 * 1024
        };

        double timings[ITERATIONS]; // This stores timings for each iteration

        for (int i = 0; i < MESSAGE_SIZES; i++) {
            int msg_size = msg_sizes[i];
            char* message = (char*)malloc(msg_size);

            double start_time, end_time;//, total_time = 0.0;

            for (int j = 0; j < ITERATIONS; j++) {
                //MPI_Barrier(MPI_COMM_WORLD);  // Synchronize processes
		//since rank 0 is sending message to rank 1, the start is computed for rank 0 
                if (j != 0 && rank == 0) {
                    // Measure time for subsequent iterations
                    start_time = MPI_Wtime();
                }

                // Communication between the nodes
                if (rank == 0) {
                    MPI_Send(message, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(message, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank == 1) {
                    MPI_Recv(message, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(message, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                }

                if (j != 0 && rank == 0) {
                    // Measure time for subsequent iterations
                    end_time = MPI_Wtime();
                    timings[j - 1] = (end_time - start_time)*1000000.0; //scaling to get output in microseconds for better precision
                }
            }

            free(message);

            if (rank == 0) {
                writeDataToCSV2(timings, msg_size, ITERATIONS, rank, size, "output_inter"); // data is written when rank is 0, for each message size
            }
        }
    }

    else if (size == 4) {
        //printf("There are 4 processes on 2 nodes \n");
        int msg_sizes[MESSAGE_SIZES] = {
            32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
            1 * 1024 * 1024, 2 * 1024 * 1024
        };

        double timings[ITERATIONS];

        for (int i = 0; i < MESSAGE_SIZES; i++) {
            int msg_size = msg_sizes[i];
            char* message = (char*)malloc(msg_size);

            double start_time, end_time;//, total_time = 0.0;

            for (int j = 0; j < ITERATIONS; j++) {
                //MPI_Barrier(MPI_COMM_WORLD);  // Synchronize processes
		// Time is noted for rank 0 and rank 1 since they belong to node 0 which is sending message to node 1	
                if (j != 0 && rank < 2 ) {
                    
                    start_time = MPI_Wtime();
                }

                // Communication between the nodes
                if (rank < 2) {
                    MPI_Send(message, msg_size, MPI_CHAR, rank + 2, 0, MPI_COMM_WORLD);
                    MPI_Recv(message, msg_size, MPI_CHAR, rank + 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(message, msg_size, MPI_CHAR, rank - 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(message, msg_size, MPI_CHAR, rank - 2, 0, MPI_COMM_WORLD);
                }

                if (j != 0 && rank < 2) {
                    end_time = MPI_Wtime();
                    timings[j - 1] = (end_time - start_time)*1000000.0;
                }
            }

            free(message);

            if (rank <2) {
                writeDataToCSV2(timings, msg_size, ITERATIONS, rank, size, "output_inter");
            }
        }
    }

    else if (size == 6) {
        //printf("There are 6 processes on 2 nodes \n");
        int msg_sizes[MESSAGE_SIZES] = {
            32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
            1 * 1024 * 1024, 2 * 1024 * 1024
        };

        double timings[ITERATIONS];

        for (int i = 0; i < MESSAGE_SIZES; i++) {
            int msg_size = msg_sizes[i];
            char* message = (char*)malloc(msg_size);

            double start_time, end_time;//, total_time = 0.0;

            for (int j = 0; j < ITERATIONS; j++) {
                //MPI_Barrier(MPI_COMM_WORLD);  // Synchronize processes
		// round trip time is computed for node 0 which has ranks from 0 till 2 	
                if (j != 0 && rank < 3 ) {
                    
                    start_time = MPI_Wtime();
                }

                // Communication between the nodes
                if (rank <3) {
                    MPI_Send(message, msg_size, MPI_CHAR, rank + 3, 0, MPI_COMM_WORLD);
                    MPI_Recv(message, msg_size, MPI_CHAR, rank + 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } 
                else {
                    MPI_Recv(message, msg_size, MPI_CHAR, rank - 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(message, msg_size, MPI_CHAR, rank - 3, 0, MPI_COMM_WORLD);
                }

                if (j != 0 && rank < 3) {
               
                    end_time = MPI_Wtime();
                    timings[j - 1] = (end_time - start_time)*1000000.0;
                }
            }

            free(message);

            if (rank <3) {
                writeDataToCSV2(timings, msg_size, ITERATIONS, rank, size, "output_inter");
            }
        }
    }

    else if (size == 8) {
        //printf("There are 8 processes on 2 nodes \n");
        int msg_sizes[MESSAGE_SIZES] = {
            32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
            1 * 1024 * 1024, 2 * 1024 * 1024
        };

        double timings[ITERATIONS];

        for (int i = 0; i < MESSAGE_SIZES; i++) {
            int msg_size = msg_sizes[i];
            char* message = (char*)malloc(msg_size);

            double start_time, end_time;//, total_time = 0.0;

            for (int j = 0; j < ITERATIONS; j++) {
                //MPI_Barrier(MPI_COMM_WORLD);  // Synchronize processes
		//Time is noted for node 0 which has ranks/processes from 0 till 3. At this point there is no difference between inter and intra comm	
                if (j != 0 && rank < 4 ) {
                    
                    start_time = MPI_Wtime();
                }

                // Communication within the nodes; process 0 and 1 of node 0 and process 4 & 5 of node 1 communicate
                if (rank % 4 == 0) {
                    MPI_Send(message, msg_size, MPI_CHAR, (rank + 1) % 8, 0, MPI_COMM_WORLD);
                    MPI_Recv(message, msg_size, MPI_CHAR, (rank + 1) % 8, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank % 4 == 1) {
                    MPI_Recv(message, msg_size, MPI_CHAR, (rank - 1) % 8, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(message, msg_size, MPI_CHAR, (rank - 1) % 8, 0, MPI_COMM_WORLD);
                }
                // Communication between the nodes; rank 2 of node 0 communicates with rank 6 of node 1and rank 3 of node 0 communicates with rank 7 of node 1 
                else if (rank ==2 || rank ==3) {
                    MPI_Send(message, msg_size, MPI_CHAR, (rank + 4) % 8, 0, MPI_COMM_WORLD);
                    MPI_Recv(message, msg_size, MPI_CHAR, (rank + 4) % 8, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else if (rank ==6 || rank ==7) {
                    MPI_Recv(message, msg_size, MPI_CHAR, (rank - 4) % 8, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(message, msg_size, MPI_CHAR, (rank - 4) % 8, 0, MPI_COMM_WORLD);
                }

                if (j != 0 && rank < 4) {
                    
                    end_time = MPI_Wtime();
                    timings[j - 1] = (end_time - start_time)*1000000.0;
                }
            }

            free(message);

            if (rank <4) {
                if(rank <2) {
		    //since rank0 and rank1 communicate within node, their rtt needs to be dumped in separate file
                    writeDataToCSV2(timings, msg_size, ITERATIONS, rank, size, "output_intra");
                }
                else {
		    //rank2 and rank3 rtt of node 0 is dummped in a separate file
                    writeDataToCSV2(timings, msg_size, ITERATIONS, rank, size, "output_inter");
                }

            }
        }
    }
    else {
        printf("The provided nodes and cores does not match any criterion. Please specify the right nodes and cores\n");
        MPI_Finalize();
        return 1;
    }

    
    MPI_Finalize();
    return 0;
}


