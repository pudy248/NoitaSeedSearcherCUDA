CC:=clang

CFLAGS:=-Iinclude -std=c++23 -O3 -march=native -fwrapv -g
#  --cuda-gpu-arch=sm_75 -L/usr/local/cuda-12.4/lib64 -lsfml-system -lsfml-graphics -lsfml-window -lcudart -ldl -lrt
LIBS:=-lpng -lstdc++ -lm
SRC:=$(wildcard src/*.cu) $(wildcard src/*.cpp) main.cu $(wildcard include/*.h)

NoitaChestFinder: $(SRC)
	$(CC) $(CFLAGS) $(LIBS) -xc++ main.cu -o NoitaChestFinder

build_profiled:
	$(CC) $(CFLAGS) $(LIBS) -fprofile-use=code.profdata -xc++ main.cu -o NoitaChestFinder

profile:
	$(CC) $(CFLAGS) $(LIBS) -fprofile-generate -xc++ main.cu -o NoitaChestFinder
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder -b coalmine -c -fi sampo -cp --end-seed 40000 --debug no_pathfinding
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder -b vault -c -i -gp -fm gold -cp --end-seed 250000 --debug no_pathfinding
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder -b crypt -c -w -gw -gs -fs nuke_giga -cp --end-seed 100000 --debug no_pathfinding
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder -b coalmine -ps -fp coalmine_oiltank_puzzle -cp --end-seed 40000 --debug no_pathfinding
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder --cart skateboard -cp --end-seed 100000000
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder --rain acid -cp --end-seed 100000000
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder --starting-flask gold -cp --end-seed 50000000
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder --alchemy {mud,water,soil} {any,any,any} -cp --end-seed 20000000
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder --biome-mods {coalmine,extremely_lucrative} -cp --end-seed 10000000
	LLVM_PROFILE_FILE="code-%p.profraw" ./NoitaChestFinder --fungal {gold,cheese_static,0,0} -cp --end-seed 5000000
	llvm-profdata merge -output=code.profdata code-*.profraw
	rm code-*.profraw NoitaChestFinder

test:
	./NoitaChestFinder -b coalmine -c -fi sampo -cp --end-seed 5000000
	./NoitaChestFinder -b vault -c -i -gp -fm gold -cp --end-seed 20000000
	./NoitaChestFinder -b crypt -c -w -gw -gs -fs nuke_giga -cp --end-seed 10000000
	./NoitaChestFinder -b coalmine -ps -fp coalmine_oiltank_puzzle -cp --end-seed 5000000
	./NoitaChestFinder --cart skateboard -cp
	./NoitaChestFinder --rain acid -cp
	./NoitaChestFinder --starting-flask gold -cp
	./NoitaChestFinder --alchemy {mud,water,soil} {any,any,any} -cp
	./NoitaChestFinder --biome-mods {coalmine,extremely_lucrative} -cp
	./NoitaChestFinder --fungal {gold,cheese_static,0,0} -cp

test2:
	@for b in wizardcave meat vault_frozen robobase; do \
	echo $$b; \
	for i in $$(seq 1 15); do\
		echo -n ">="$$i": ";\
		./NoitaChestFinder --debug no_pathfinding -b $$b -i -cp --end-seed 20000000 -a -fi "{"heart,$$i"}" -li 0;\
    done;\
	done;