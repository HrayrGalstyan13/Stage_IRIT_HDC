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

Compile:

```bash
gcc main.c -o language_detection -lcsv -lm
```

### MNIST Project

Required:

* Standard C library
* math library

Compile:

```bash
gcc main.c -o mnist -lm
```

## Dataset Files

### Language Detection

Place `language_detection.csv` in the project directory.

### MNIST

Place the following files in the project directory:

* train-images.idx3-ubyte
* train-labels.idx1-ubyte
* t10k-images.idx3-ubyte
* t10k-labels.idx1-ubyte

```
```
