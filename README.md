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

# Python Projects – How to Run

### Install dependencies

```bash
pip install -r requirements_py.txt
```

---

### MNIST Projects

```bash
cd mnist_py
python main.py
```

```bash
cd le_mnist_py
python main.py
```

```bash
cd local_mnist_py
python main.py
```

---

### Language Detection

```bash
cd language_py
python main.py
```
# Dataset Download (If Missing)

If the datasets are not present in the project directory, download them from the following sources:

### Language Dataset

https://www.kaggle.com/basilb2s/datasets

### MNIST Dataset

https://www.kaggle.com/datasets/hojjatk/mnist-dataset

### Alternative Dataset (Optional)

You can also use **Fashion-MNIST** instead of MNIST. It has the same format and file structure:

https://www.kaggle.com/datasets/zalando-research/fashionmnist

Make sure the following files are placed in the correct folders before running:

* `train-images.idx3-ubyte`
* `train-labels.idx1-ubyte`
* `t10k-images.idx3-ubyte`
* `t10k-labels.idx1-ubyte`
