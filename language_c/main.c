#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>  
#include <sys/resource.h>
#include <sys/types.h>
#include <csv.h>     

#define DIM 10000              
#define MAX_CLASSES 50
#define MAX_SAMPLES 20000
#define NGRAM 3


// Data Structures

char *texts[MAX_SAMPLES];
int labels[MAX_SAMPLES];
int num_samples = 0;

char current_text[1024];
char current_lang[50];
int field_index = 0;

char class_names[MAX_CLASSES][50];
int num_classes = 0;

// Hypervectors
int char_memory[256][DIM];
int class_hv[MAX_CLASSES][DIM];


// Hypervector Operations


// Initialize random hypervectors for characters (binding and permutation)
void init_char_memory() {
    for (int c = 0; c < 256; c++) {
        for (int i = 0; i < DIM; i++) {
            char_memory[c][i] = (rand() % 2) ? 1 : -1;
        }
    }
}

// Binding operation (element-wise multiplication)
void bind(int *hv1, int *hv2, int *out) {
    for (int i = 0; i < DIM; i++) {
        out[i] = hv1[i] * hv2[i]; // Binding using element-wise multiplication
    }
}

// Permutation operation (shuffle the hypervector)
void permute(int *hv, int *out) {
    for (int i = 0; i < DIM; i++) {
        out[i] = hv[(i + 1) % DIM]; // Simple permutation (shift right by 1)
    }
}

// Bundling operation (element-wise addition followed by binarization)
void bundle(int *hv1, int *hv2, int *out) {
    for (int i = 0; i < DIM; i++) {
        out[i] = hv1[i] + hv2[i]; // Bundling by addition
    }
}



// Encode text → hypervector
void encode_text(const char *text, int *hv) {
    int len = strlen(text);

    
    for (int i = 0; i < DIM; i++) hv[i] = 0;

    for (int i = 0; i < len - NGRAM + 1; i++) {

        int temp[DIM];
        for (int d = 0; d < DIM; d++) temp[d] = 1;

        int permuted[DIM];

        //Apply permutation for position encoding
        for (int j = 0; j < NGRAM; j++) {
            unsigned char ch = text[i + j];

            // Copy char memory
            for (int d = 0; d < DIM; d++) {
                permuted[d] = char_memory[ch][d];
            }

            // Apply j permutations
            for (int p = 0; p < j; p++) {
                int tmp[DIM];
                permute(permuted, tmp);
                memcpy(permuted, tmp, sizeof(int) * DIM);
            }

            // Bind
            for (int d = 0; d < DIM; d++) {
                temp[d] *= permuted[d];
            }
        }

        // Accumulate (no binarization)
        for (int d = 0; d < DIM; d++) {
            hv[d] += temp[d];
        }
    }

    //FINAL BINARIZATION
    for (int i = 0; i < DIM; i++) {
        hv[i] = (hv[i] >= 0) ? 1 : -1;
    }
}

// Cosine similarity (not used here)

double cosine(int *a, int *b) {
    double dot = 0, na = 0, nb = 0;

    for (int i = 0; i < DIM; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    return dot / (sqrt(na) * sqrt(nb) + 1e-8);
}

double hamming_similarity(int *a, int *b) {
    int hamming_distance = 0;

    // Iterate through the vectors and count the differing positions
    for (int i = 0; i < DIM; i++) {
        if (a[i] != b[i]) {
            hamming_distance++;
        }
    }

    // Calculate similarity: 1 - (Hamming Distance / DIM)
    return 1.0 - (double)hamming_distance/ DIM;
}

// Predict language from text

int predict(const char *text) {
    int hv[DIM];
    encode_text(text, hv);
    if (text == NULL) {
        printf("NULL\n");
        //return -2;  // Error code -2: NULL input text
    }

    int len = strlen(text);
    if (len == 0) {
        printf("len\n");
        //return -1;  // Error code -1: Empty text
    }

    double best = -1e9;
    int best_class = -1;

    for (int c = 0; c < num_classes; c++) {
        double sim = hamming_similarity(hv, class_hv[c]);
        if (sim >= best) {
            best = sim;
            best_class = c;
        }
    }
    return best_class;
}


// Callback functions for CSV parsing

void cb_field(void *field, size_t len, void *data) {
    char *f = (char *)field;
    if (field_index == 0) {
        snprintf(current_text, sizeof(current_text), "%.*s", (int)len, f);
    } else if (field_index == 1) {
        snprintf(current_lang, sizeof(current_lang), "%.*s", (int)len, f);
    }

    field_index++;
}

void cb_row(int c, void *data) {
    if (field_index == 2) {  // Expect exactly 2 columns
        if (strlen(current_text) > 0 && strlen(current_lang) > 0) {
            texts[num_samples] = strdup(current_text);

            int label = -1;
            for (int i = 0; i < num_classes; i++) {
                if (strcmp(class_names[i], current_lang) == 0) {
                    label = i;
                    break;
                }
            }

            if (label == -1) {
                label = num_classes;
                strcpy(class_names[num_classes], current_lang);
                num_classes++;
            }

            labels[num_samples] = label;
            num_samples++;
        }
    }
    field_index = 0; // Reset for next row
}


// Load data (preserving the original version with CSV parsing)

void load_data(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open file");
        exit(1);
    }

    struct csv_parser parser;

    if (csv_init(&parser, CSV_STRICT) != 0) {
        fprintf(stderr, "Failed to init csv parser\n");
        fclose(fp);
        exit(1);
    }

    char buffer[1024];
    size_t bytes;

    int is_header = 1;

    while ((bytes = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        if (is_header) {
            // Skip header manually by consuming first row
            char *newline = memchr(buffer, '\n', bytes);
            if (newline) {
                size_t offset = (newline - buffer) + 1;
                csv_parse(&parser, buffer + offset, bytes - offset,
                          cb_field, cb_row, NULL);
                is_header = 0;
            }
        } else {
            if (csv_parse(&parser, buffer, bytes,
                          cb_field, cb_row, NULL) != bytes) {
                fprintf(stderr, "CSV parse error: %s\n",
                        csv_strerror(csv_error(&parser)));
                exit(1);
            }
        }
    }

    csv_fini(&parser, cb_field, cb_row, NULL);
    csv_free(&parser);

    fclose(fp);
}


// CPU/RAM usage tracking (Linux-specific)

void track_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    // CPU Usage
    long cpu_time = usage.ru_utime.tv_sec * 1000000 + usage.ru_utime.tv_usec;
    printf("CPU Usage: %ld microseconds\n", cpu_time);

    // Memory Usage in GB
    double memory_usage_gb = (double)usage.ru_maxrss / (1024.0 * 1024.0);
    printf("Memory Usage: %.2f GB\n", memory_usage_gb);

    // Track CPU usage by parsing /proc/stat
    FILE *fp = fopen("/proc/stat", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "cpu", 3) == 0) {
                // Print the CPU stats from /proc/stat (including usage per core)
                printf("CPU Stats: %s", line);
            }
        }
        fclose(fp);
    } else {
        perror("Failed to open /proc/stat");
    }
}


// Track CPU/RAM usage for Training and Testing Separately




// Confusion Matrix & Accuracy

void print_confusion_matrix(int *y_true, int *y_pred, int n) {
    int cm[MAX_CLASSES][MAX_CLASSES] = {0};

    for (int i = 0; i < n; i++) {
        cm[y_true[i]][y_pred[i]]++;
    }

    // Print the language names above the confusion matrix
    printf("\nConfusion Matrix\n");
    printf("%-15s", " ");
    for (int i = 0; i < num_classes; i++) {
        printf("%-10s", class_names[i]);
    }
    printf("\n");

    for (int i = 0; i < num_classes; i++) {
        printf("%-15s", class_names[i]);
        for (int j = 0; j < num_classes; j++) {
            printf("%4d ", cm[i][j]);
        }
        printf("\n");
    }

    // Calculate Accuracy
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    double accuracy = (double)correct / n * 100;
    printf("Accuracy: %.2f%%\n", accuracy);
}


// Shuffle the indices

void shuffle_data(int *indices, int n) {
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    for (int i = 0; i < n; i++) {
        int j = rand() % n;
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}


// Main function

int main() {
    srand(time(NULL));

    // Track start time
    clock_t start_time = clock();

    // Load data
    printf("Loading data...\n");
    load_data("language_detection.csv");

    // Initialize character hypervectors
    init_char_memory();

    int indices[MAX_SAMPLES];
    shuffle_data(indices, num_samples);  // Shuffle data indices

    int split = (int)(0.8 * num_samples);


    // Training phase

    printf("Training...\n");
    clock_t start_training = clock();
    for (int i = 0; i < num_classes; i++) {
        for (int d = 0; d < DIM; d++) {
            class_hv[i][d] = 1;  // Initialize class hypervectors
        }
    }

    for (int i = 0; i < split; i++) {
        int hv[DIM];
        encode_text(texts[indices[i]], hv);

        int label = labels[indices[i]];
        for (int d = 0; d < DIM; d++) {
            class_hv[label][d] += hv[d];
        }

        // Print training sample every 500 samples
        /*if (i % 500 == 0) {
            printf("Training Sample: Language: %s, Text: %s\n", class_names[labels[indices[i]]], texts[indices[i]]);
            printf("HV (first 20 dims): ");
            for (int d = 0; d < 20; d++) {
                printf("%d ", hv[d]);
            }
            printf("\n");
        }*/
    }

    // Binarize class hypervectors
    for (int c = 0; c < num_classes; c++) {
        for (int d = 0; d < DIM; d++) {
            class_hv[c][d] = (class_hv[c][d] >= 0) ? 1 : -1;
        }
    }
    clock_t end_training = clock();
    


    // Testing phase

    printf("Testing...\n");
    clock_t start_testing = clock();
    int y_true[MAX_SAMPLES];
    int y_pred[MAX_SAMPLES];
    int test_size = 0;

    for (int i = split; i < num_samples; i++) {
        y_true[test_size] = labels[indices[i]];
        y_pred[test_size] = predict(texts[indices[i]]);
        test_size++;
        /*if (test_size % 50 == 0) {
        printf("Test Sample: Language: %s, Text: %s\n", class_names[y_true[test_size - 1]], texts[indices[i]]);
        int hv_test[DIM];
        encode_text(texts[indices[i]], hv_test);

        printf("HV (first 20 dims): ");
        for (int d = 0; d < 20; d++) {
            printf("%d ", hv_test[d]);
        }
        printf("\n");
    }*/
    }
    clock_t end_testing = clock();
    

    // Evaluation metrics

    print_confusion_matrix(y_true, y_pred, test_size);

    // Track CPU and memory usage
    track_usage();

    // Track end time
    clock_t end_time = clock();
    double training_seconds = (double)(end_training - start_training) / CLOCKS_PER_SEC;
    double testing_seconds = (double)(end_testing - start_testing) / CLOCKS_PER_SEC;
    double runtime_seconds = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Total Training: %.2f seconds\n", training_seconds);
    printf("Total Testing: %.2f seconds\n", testing_seconds);
    printf("Total Runtime: %.2f seconds\n", runtime_seconds);

    return 0;
}