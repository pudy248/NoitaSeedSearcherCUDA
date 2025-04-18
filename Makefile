CC:=clang

CFLAGS:=-Iinclude -std=c++23 -O3 -march=native -fwrapv --cuda-gpu-arch=sm_75 -Wno-enum-compare -g -fsanitize=memory -fsanitize-memory-track-origins
LIBS:=-L/usr/local/cuda-12.4/lib64 -lpng -lsfml-system -lsfml-graphics -lsfml-window -lstdc++ -lm -lcudart -ldl -lrt
SRC:=$(wildcard src/*.cu) $(wildcard src/*.cpp) main.cu $(wildcard include/*.h)

main: $(SRC)
	$(CC) $(CFLAGS) $(LIBS) main.cu -o main