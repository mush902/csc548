#ifndef MY_MPI_H_
#define MY_MPI_H_
#include <stdio.h>
#include <stdlib.h>
#define MPI_Datatype int
#define MPI_DOUBLE 8
#define MPI_INT 4
#define MPI_CHAR 1
#define MPI_Status int
#define MPI_STATUS_IGNORE 0

typedef struct {
    int port;
    char n_name[5];
} Node;

typedef struct {
    int rank;
    int size;
    int port;
    int sockfd;
    char *nodename;
    Node *Nodes;
} MPI_Comm;

extern MPI_Comm MPI_COMM_WORLD;

int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Init(int *argc, char **argv[]);
int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Finalize();
double MPI_Wtime();

#endif

