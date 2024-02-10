CC := nvcc
CFLAGS := -O3 -lm -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include
LDFLAGS := -L$(CUDA_HOME)/lib64
TARGET := p2
SRC := p2.cu

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(TARGET)

