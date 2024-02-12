# Compiler 
CC = gcc

# Compile flags
CFLAGS = -Wall -g

# Libraries
LIBS = 

# Target executable
TARGET = my_rtt

# Object files 
OBJS = my_mpi.o my_rtt.o

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)

my_mpi.o: my_mpi.c my_mpi.h
	$(CC) -c $(CFLAGS) $<

my_rtt.o: my_rtt.c my_mpi.h
	$(CC) -c $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
