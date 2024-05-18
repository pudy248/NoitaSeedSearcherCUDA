CC:=clang-17

CFLAGS:=-Iinclude -std=c++17 -Ofast --cuda-gpu-arch=sm_75  -Wno-enum-compare
LIBS:=-L/usr/local/cuda-12.4/lib64 -lpng -lsfml-system -lsfml-graphics -lsfml-window -lstdc++ -lm -lcudart -ldl -lrt
SRC:=$(wildcard src/*.cu) $(wildcard src/*.cpp) main.cu $(wildcard include/*.h)

main: $(SRC)
	$(CC) $(CFLAGS) $(LIBS) main.cu -o main