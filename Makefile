NVCC := nvcc
target := histogram_app
SRC := main.cu histogram_kernel.cu
OBJ := $(SRC:.cu=.o)

all: $(target)

$(target): $(OBJ)
	$(NVCC) -o $@ $^ --std=c++11

%.o: %.cu histogram_kernel.h
	$(NVCC) -c $< -o $@

clean:
	rm -f *.o $(target)