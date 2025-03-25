CC:=clang

CFLAGS:=-Iinclude -std=c++23 -O3 --cuda-gpu-arch=sm_75  -Wno-enum-compare
LIBS:=-L/usr/local/cuda-12.4/lib64 -lpng -lsfml-system -lsfml-graphics -lsfml-window -lstdc++ -lm -lcudart -ldl -lrt
SRC:=$(wildcard src/*.cu) $(wildcard src/*.cpp) main.cu

main: $(SRC)
	$(CC) $(CFLAGS) $(LIBS) main.cu -o main