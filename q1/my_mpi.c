#include "my_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/types.h>
#include <netdb.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>

MPI_Comm MPI_COMM_WORLD;

void error(const char *msg) {
    perror(msg);
    exit(1);
}

int create_socket(int port) {
    int sockfd = -1;
    struct sockaddr_in serv_addr;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("Error: Failed to open socket");
    int optval = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0)
        error("Error: setsockopt failed");
    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0)
        error("Error: setsockopt timeout");
    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(port);
    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        error("Error: Failed to bind at sockaddr");
    listen(sockfd, 5);
    return sockfd; // returns the socket file descriptor for communication
}

// allows creation of socket connections from process calling the function to another host at defined port
int connect_client(char *hostname, int port) {
    int sockfd;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("Error: Failed to open the socket");
    server = gethostbyname(hostname);
    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(port);
    while (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0);
    return sockfd; // returns client file descriptor for communication
}


int MPI_Init(int *argc, char **argv[]) {
    printf("In MPI_Init \n");
    MPI_COMM_WORLD.rank = atoi(getenv("MYMPI_RANK"));
    MPI_COMM_WORLD.size = atoi(getenv("MYMPI_NTASKS"));
    MPI_COMM_WORLD.nodename = getenv("MYMPI_NODENAME");
    char *source_node = getenv("MYMPI_SOURCENODE");
    int sockfd = create_socket(1024 + MPI_COMM_WORLD.rank);
    if (sockfd < 0)
        error("Error: Failed to create the socket");
    MPI_COMM_WORLD.port = 1024 + MPI_COMM_WORLD.rank;
    MPI_COMM_WORLD.sockfd = sockfd;
    MPI_COMM_WORLD.Nodes = (Node *)malloc(MPI_COMM_WORLD.size * sizeof(Node));
    char message[20] = "", gather_read[200] = "" ;
    sprintf(message, "%s,%d,%d", MPI_COMM_WORLD.nodename, MPI_COMM_WORLD.rank, MPI_COMM_WORLD.port);
    //printf("Before message exchanges \n");
    if (MPI_COMM_WORLD.rank == 0) {
        for (int i = 1; i < MPI_COMM_WORLD.size; i++) {
            socklen_t clilen;
            struct sockaddr_in cli_addr;
            clilen = sizeof(cli_addr);
            char buffer[1024];
            int flags = fcntl(MPI_COMM_WORLD.sockfd, F_GETFL, 0);
            fcntl(MPI_COMM_WORLD.sockfd, F_SETFL, flags & ~O_NONBLOCK);
            int newsockfd;
            while ((newsockfd = accept(MPI_COMM_WORLD.sockfd, (struct sockaddr *)&cli_addr, &clilen)) < 0) {
                if (errno == EINTR) {
                    continue; // Retry the accept
                } else {
                    error("Error: Failed to accept connection");
                    break; // Exit the loop on non-EINTR errors
                }
            }
            bzero(buffer, 1024);
            int n = read(newsockfd, buffer, 1023); // read the message into buffer
            if (n < 0)
                error("Error: Failed to read from the socket");
            char *msg = (char *)malloc(n);
            memcpy(msg, buffer, n);
            close(newsockfd);
            strcat(gather_read, msg);
            strcat(gather_read, "\n");
            free(msg);
        }
        strcat(gather_read, message);
    } else {
        int clientfd = connect_client(source_node, 1024);
        write(clientfd, message, strlen(message)); 
        close(clientfd);
    }
    //printf("After message exchange 1 \n");
    if (MPI_COMM_WORLD.rank == 0) {
        char gather_write[500];
        strcpy(gather_write, gather_read);
        char *token;
        token = strtok(gather_write, "\n,");
        for(; token != NULL; token = strtok(NULL, "\n,")) {
            char *host = token; 
            token = strtok(NULL, "\n,");
            int rank = atoi(token);
            token = strtok(NULL, "\n,");  
            int port = atoi(token);
            strcpy(MPI_COMM_WORLD.Nodes[rank].n_name + strlen(MPI_COMM_WORLD.Nodes[rank].n_name), host);
            MPI_COMM_WORLD.Nodes[rank].port = port; 
        }
        for (int i = 1; i < MPI_COMM_WORLD.size; i++) {
            int clientfd = connect_client(MPI_COMM_WORLD.Nodes[i].n_name, MPI_COMM_WORLD.Nodes[i].port);
            write(clientfd, gather_read, strlen(gather_read)); 
            close(clientfd);
        }
    } else {
        socklen_t clilen;
        struct sockaddr_in cli_addr;
        clilen = sizeof(cli_addr);
        char buffer[1024];
        int flags = fcntl(MPI_COMM_WORLD.sockfd, F_GETFL, 0);
        fcntl(MPI_COMM_WORLD.sockfd, F_SETFL, flags & ~O_NONBLOCK);
        int newsockfd;
        while ((newsockfd = accept(MPI_COMM_WORLD.sockfd, (struct sockaddr *)&cli_addr, &clilen)) < 0) {
            if (errno == EINTR) {
                continue; // Retry the accept
            } else {
                error("Error: Failed to accept connection");
                break; // Exit the loop on non-EINTR errors
            }
        }
        bzero(buffer, 1024);
        int n = read(newsockfd, buffer, 1023); // read the message into buffer
        if (n < 0)
            error("Error: Failed to read from the socket");
        char *msg = (char *)malloc(n);
        memcpy(msg, buffer, n);
        close(newsockfd);
        char *token;
        token = strtok(msg, "\n,");
        for(; token != NULL; token = strtok(NULL, "\n,")) {
            char *host = token; 
            token = strtok(NULL, "\n,");
            int rank = atoi(token);
            token = strtok(NULL, "\n,");  
            int port = atoi(token);
            strcpy(MPI_COMM_WORLD.Nodes[rank].n_name + strlen(MPI_COMM_WORLD.Nodes[rank].n_name), host);
            MPI_COMM_WORLD.Nodes[rank].port = port; 
        }
        free(msg);
    }
    printf("After message exchange 2 \n");
    sleep(3);
    return 0;
}


int MPI_Send(void *buffer, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    char *host = comm.Nodes[dest].n_name;
    int portno = comm.Nodes[dest].port;
    int clientfd = connect_client(host, portno);
    int total = 0;
    //printf("In MPI_Send portno %d clientfd %d \n", portno, clientfd);
    while (total < count * datatype) {
        int n = write(clientfd, buffer + total, count * datatype);
        if (n < 0)
            perror("Error: Failed to write in MPI_Send");
        if (n == 0)
            break;
        total += n;
    }
    close(clientfd);
    return (0);
}

int MPI_Recv(void *buffer, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {
    socklen_t clilen;
    struct sockaddr_in cli_addr;
    clilen = sizeof(cli_addr);
    int newsockfd = accept(comm.sockfd, (struct sockaddr *)&cli_addr, &clilen);
    if (newsockfd < 0)
        error("Error: Failed to accept connection");
    int total = 0;
    while (total < count * datatype) {
        int n = read(newsockfd, buffer + total, count * datatype);
        if (n < 0)
            perror("Error: Failed to read in MPI_Recv");
        total += n;
        if (n == 0)
            break;
    }
    close(newsockfd);
    return 0;
}

int MPI_Comm_size(MPI_Comm comm, int *size) {
    *size = comm.size;
    return 0;
}

int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    *rank = comm.rank;
    return 0;
}

int MPI_Finalize() {
    close(MPI_COMM_WORLD.sockfd);
    return 0;
}

double MPI_Wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)(tv.tv_sec * 1e6 + tv.tv_usec) * 1e-6;
}

