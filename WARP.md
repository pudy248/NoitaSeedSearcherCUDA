# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: NoitaSeedSearcherCUDA (binary: NoitaChestFinder)

Scope: High-signal commands for building/running and a big-picture architecture overview to ramp quickly.

Common development commands

- Build on Windows (Visual Studio solution)

```powershell path=null start=null
# Build the Visual Studio solution in Release configuration
# - Requires the "Developer Command Prompt for VS" (or VS Build Tools) so msbuild is available
# - Produces the NoitaChestFinder binary under the solution's output directory (e.g., x64/Release)
msbuild .\NoitaChestFinderCuda_2.sln /p:Configuration=Release /m

# Build in Debug configuration (useful for local debugging)
msbuild .\NoitaChestFinderCuda_2.sln /p:Configuration=Debug /m
```

- Build with Makefile (Unix-like environments or Windows with make/clang available)

```bash path=null start=null
# Build the NoitaChestFinder binary using clang
# - Requires clang, libpng, and standard C++ libs on your system
# - Current Makefile compiles CPU code via main.cu with -xc++ (CUDA libs are commented out)
make -j

# Optional: Profile-Guided Optimization (PGO) flow using LLVM tools
# 1) Build an instrumented binary and run representative workloads to collect .profraw files
make profile
# 2) Merge profiles into code.profdata, then build with -fprofile-use
make build_profiled
```

- Run the program

```powershell path=null start=null
# Show CLI help with all supported flags and descriptions
# (This mirrors README.md: "run NoitaChestFinder --help" for usage overview)
.\NoitaChestFinder --help

# Example run: search coalmine for chests/heart with sampo, counting passes only to speed up
# -b coalmine  => generate coalmine biome
# -c           => include biome chests
# -fi sampo    => filter items for 'sampo'
# -cp          => count matches only (do not record seeds)
# --end-seed   => set last seed to search (start defaults to 1)
.\NoitaChestFinder -b coalmine -c -fi sampo -cp --end-seed 5000000
```

- Smoke checks (no unit test framework present; Makefile provides scenario runs)

```bash path=null start=null
# Run a curated set of scenarios to sanity-check functionality
make test

# Run the parametric 'test2' scenario for a specific biome (GNU make syntax)
make test2 BIOME=crypt
```

```powershell path=null start=null
# Windows equivalent of a single 'test2' line without make (adjust BIOME/filters as needed)
# This example searches the 'crypt' biome with specific filters and prints only completed runs
.\NoitaChestFinder --debug no_pathfinding -b crypt -i -cp --end-seed 20000000 -a -fi "{heart,10}" | Select-String finished
```

- Useful CLI patterns (compose your own single-run checks)

```powershell path=null start=null
# Explore pixel-scene puzzles in coalmine oil tank
.\NoitaChestFinder -b coalmine -ps -fp coalmine_oiltank_puzzle -cp --end-seed 40000

# Shop spells and wands in vault (requires --gen-spells/--gen-wands for detailed generation)
.\NoitaChestFinder -b vault -c -i -gp -fm gold -cp --end-seed 250000

# Alchemy: specify LC/AP as composites (ordering optional)
.\NoitaChestFinder --alchemy "{mud,water,soil}" "{any,any,any}" -cp --end-seed 20000000
```

Architecture overview (big picture)

- Entry point and build pattern
  - main.cu is the entry point and directly includes most implementation units (src/*.cpp), the platform implementation (platforms/platform_implementation.h and src/platform_implementation_src.cpp), and PNG utilities (include/pngutils.h).
  - This compile-time inclusion yields a single translation unit and simplifies cross-backend optimizations.

- Platform abstraction (CPU and CUDA backends)
  - Public API: platforms/platform_api.h exposes functions like InitializePlatform, UploadToDevice, SetTargetDispatchRate, DestroyPlatform, etc.
  - Implementations: platforms/impl_cpu.* and platforms/impl_cuda.* provide concrete backends. The platform_implementation glue selects appropriate paths and helpers in platforms/platform_* and *_compute_helpers.*.
  - main.cu initializes the platform, configures dispatch rate (when DEBUG_DISPATCH_RATE_OVERRIDE is set), allocates device/host buffers, runs the search, and tears down the platform.

- CLI and configuration
  - src/cli.cpp defines a table-driven command system with long/short flags and rich parsing for composite arguments (e.g., {from,to[,min,max]}).
  - Flags map directly into a global config object with sections:
    - generalCfg: seed range and scheduling (seedStart, seedEnd, seedBlockSize, priority).
    - precheckCfg: one-time filters before biome generation (cart, flask, rain, alchemy, biome modifiers, fungal, perks, upwarps).
    - spawnableCfg: per-biome generation toggles (biomeChests, pedestals, altars, pixel-scene indexing/search, shop items, etc.).
    - filterCfg: post-generation filters (items/materials/spells/pixel-scenes/wand stats, and aggregation mode).
    - outputCfg: mode (text or image), output file/path, progress interval, quiet/verbose switches.
  - The help menu is generated from the command table; run the binary with --help to see authoritative documentation.

- World generation and search pipeline
  - Tables: data/*.h encodes IDs and host-side tables for materials, perks, modifiers, spells, etc.; include/noita_random.h models PRNG behavior.
  - Biome map and instantiation: biome data is loaded (e.g., load_biome_map("data/biome_impl/biome_map.png", 0)), then InstantiateBiomes builds biome scopes used by the generator.
  - Spell data preparation: GenerateSpellData constructs probability tables and uploads compact representations to the active platform (CPU/CUDA).
  - Memory sizing: config.memSizes is derived from biome counts and map area to size buffers for outputs, map data, visited flags, and spawnables.
  - Search execution: AllocateComputeMemory → SearchMain (core kernel/loop) → FreeComputeMemory, with progress/output handled per outputCfg.

- GUI hooks (optional)
  - src/gui.* directories and SFML-related linker flags are present but currently disabled/commented (see Makefile and main.cu). The CLI is the primary interface.

Repository-specific notes

- Windows vs. Makefile: Prefer the Visual Studio solution on Windows for CUDA-enabled development. The Makefile uses clang and currently targets CPU code unless you customize CUDA flags and link libraries.
- Working directory: main sets the current directory to the executable path; run the binary from its folder so data assets (e.g., data/biome_impl/biome_map.png) are found by relative paths.
- Fast iterations: When exploring large seed ranges, use --count-passed to avoid recording each seed; combine with --quiet and --logging-interval 0 to reduce I/O overhead when appropriate.
