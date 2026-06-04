#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define DIM 10000
#define NUM_CLASSES 10
#define IMAGE_SIZE 784   //28x28
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

//Hypervectors
int8_t pixel_memory[8][DIM];     //8 bins (0–7)
int class_hv[NUM_CLASSES][DIM];

//Generate random ±1
int8_t rand_pm_one() {
    return (rand() % 2) ? 1 : -1;
}

int read_int(FILE *f) {
    unsigned char bytes[4];
    size_t n = fread(bytes, 1, 4, f);
    if (n != 4) {
        fprintf(stderr, "Error reading int from file\n");
        exit(1);
    }
    return (bytes[0]<<24) | (bytes[1]<<16) | (bytes[2]<<8) | bytes[3];
}

void load_labels(const char *filename, uint8_t *labels, int expected_size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("Error opening %s\n", filename);
        exit(1);
    }

    int magic = read_int(f);
    int size  = read_int(f);

    if (magic != 2049) {
        printf("Invalid label file!\n");
        exit(1);
    }

    if (size != expected_size) {
        printf("Size mismatch in labels!\n");
        exit(1);
    }

    size_t n = fread(labels, 1, size, f);
    if (n != size) {
        fprintf(stderr, "Error reading labels\n");
        exit(1);
    }
    fclose(f);
}

void load_images(const char *filename,
                 uint8_t images[][IMAGE_SIZE],
                 int expected_size) {

    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("Error opening %s\n", filename);
        exit(1);
    }

    int magic = read_int(f);
    int size  = read_int(f);
    int rows  = read_int(f);
    int cols  = read_int(f);

    if (magic != 2051) {
        printf("Invalid image file!\n");
        exit(1);
    }

    if (size != expected_size || rows*cols != IMAGE_SIZE) {
        printf("Image size mismatch!\n");
        exit(1);
    }

    size_t n = fread(images, 1, size * IMAGE_SIZE, f);
        if (n != size * IMAGE_SIZE) {
    fprintf(stderr, "Error reading images\n");
    exit(1);
        }
    fclose(f);
}

//Initialize pixel memory
void init_pixel_memory() {
    for (int b = 0; b < 8; b++) {
        for (int i = 0; i < DIM; i++) {
            pixel_memory[b][i] = rand_pm_one();
        }
    }
}

//Encode image → hypervector
void encode_image(uint8_t *image, int *hv) {
    for (int i = 0; i < DIM; i++) hv[i] = 0;

    for (int i = 0; i < IMAGE_SIZE; i++) {
        int bin = image[i] / 32;  // 0–7

        for (int d = 0; d < DIM; d++) {
            int shifted = (d + i) % DIM;
            hv[d] += pixel_memory[bin][shifted];
        }
    }

    //Sign activation
    for (int i = 0; i < DIM; i++) {
        hv[i] = (hv[i] >= 0) ? 1 : -1;
    }
}

//Train
void train(uint8_t images[TRAIN_SIZE][IMAGE_SIZE],
           uint8_t labels[TRAIN_SIZE]) {

    //Initialize class vectors
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int d = 0; d < DIM; d++)
            class_hv[c][d] = 0;

    int hv[DIM];

    for (int i = 0; i < TRAIN_SIZE; i++) {
        encode_image(images[i], hv);
        int label = labels[i];

        for (int d = 0; d < DIM; d++) {
            class_hv[label][d] += hv[d];
        }
    }

    //Final sign
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int d = 0; d < DIM; d++) {
            class_hv[c][d] = (class_hv[c][d] >= 0) ? 1 : -1;
        }
    }
}

//Cosine similarity
float cosine_similarity(int *a, int *b) {
    int dot = 0;
    int norm_a = 0;
    int norm_b = 0;

    for (int i = 0; i < DIM; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    return dot / (sqrt(norm_a) * sqrt(norm_b) + 1e-8);
}

//Predict
int predict(uint8_t *image) {
    int hv[DIM];
    encode_image(image, hv);

    float best_sim = -1e9;
    int best_class = 0;

    for (int c = 0; c < NUM_CLASSES; c++) {
        float sim = cosine_similarity(class_hv[c], hv);
        if (sim > best_sim) {
            best_sim = sim;
            best_class = c;
        }
    }

    return best_class;
}

//Evaluate
float test(uint8_t images[TEST_SIZE][IMAGE_SIZE],
           uint8_t labels[TEST_SIZE]) {

    int correct = 0;

    for (int i = 0; i < TEST_SIZE; i++) {
        int pred = predict(images[i]);
        if (pred == labels[i]) correct++;
    }

    return (float)correct / TEST_SIZE;
}

//MAIN
int main() {
    srand(42);

    static uint8_t train_images[TRAIN_SIZE][IMAGE_SIZE];
    static uint8_t train_labels[TRAIN_SIZE];
    static uint8_t test_images[TEST_SIZE][IMAGE_SIZE];
    static uint8_t test_labels[TEST_SIZE];

    //Load MNIST
    load_images("train-images.idx3-ubyte", train_images, TRAIN_SIZE);
    load_labels("train-labels.idx1-ubyte", train_labels, TRAIN_SIZE);

    load_images("t10k-images.idx3-ubyte", test_images, TEST_SIZE);
    load_labels("t10k-labels.idx1-ubyte", test_labels, TEST_SIZE);

    init_pixel_memory();

    clock_t t1 = clock();
    train(train_images, train_labels);
    clock_t t2 = clock();

    float acc = test(test_images, test_labels);
    clock_t t3 = clock();

    printf("Accuracy: %.4f\n", acc);
    printf("Train time: %.2f sec\n", (float)(t2 - t1) / CLOCKS_PER_SEC);
    printf("Test time: %.2f sec\n", (float)(t3 - t2) / CLOCKS_PER_SEC);
    /*int pred_count[10] = {0};
for (int i = 0; i < TEST_SIZE; i++) {
    int p = predict(test_images[i]);
    pred_count[p]++;
}
for (int i = 0; i < 10; i++) {
    printf("Pred %d: %d\n", i, pred_count[i]);
}*/

    return 0;
}