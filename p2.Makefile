#Group info:
#mshaikh2 Mushtaq Ahmed Shaikh
#ahgada Amay Gada
CC = mpicc

p2make: p2_mpi.o
	$(CC) -lm -O3 -o p2_mpi p2_mpi.c p2_func.c

