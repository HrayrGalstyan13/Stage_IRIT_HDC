# C Project Requirements

## Compiler

* GCC 11+ (recommended)

## Libraries

### Language Detection Project

Required:

* libcsv

Ubuntu/WSL installation:

```bash
sudo apt update
sudo apt install libcsv-dev
```

Compile (standard):

```bash
gcc main.c -o language_detection -lcsv -lm
```

Compile (optimized):

```bash
gcc -O3 -march=native main.c -o language_detection -lcsv -lm
```

### MNIST Project

Required:

* Standard C library
* math library (`libm`)

Compile (standard):

```bash
gcc main.c -o mnist -lm
```

Compile (optimized):

```bash
gcc -O3 -march=native main.c -o mnist -lm
```

## Dataset Files

### Language Detection

Place:

```text
language_detection.csv
```

in the project directory.

### MNIST

Place the following files in the project directory:

```text
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

## Notes

`-O3` enables aggressive compiler optimizations.

`-march=native` allows GCC to generate code optimized for the CPU on which it is compiled. This can significantly improve performance but may reduce portability to older or different processor architectures.
