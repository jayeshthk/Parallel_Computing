# Matrix multiplication is critical for language models. this code simulates the MM.

## info:

- matrixes are hardcoded into .cu file in order to change it you can edit `matrix_multiply.cu` file.

```cpp
    int M = 3;  // Rows in A and C
    int N = 3;  // Columns in A and Rows in B
    int P = 3;  // Columns in B and C

    int A[M * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[N * P] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int C[M * P]; // Result matrix
```

### How to Compile and Run:

To compile and run this CUDA program, you need to have NVIDIA's CUDA toolkit installed and a compatible GPU. Here are the steps:

1. **Compile the code** using NVIDIA's `nvcc` compiler. Open a terminal and run the following command:

   ```bash
   nvcc matrix_multiply.cu -o matrix_multiply
   ```

   This will create an executable file named `matrix_multiply`.

2. **Run the program**:

   ```bash
   ./matrix_multiply
   ```

   You should see the result of the matrix multiplication printed on the terminal.

### Example Output:

If you run the above program, it should print something like this:

```
Matrix C (Result):
30 24 18
84 69 54
138 114 90
```

This indicates that the multiplication of matrices `A` and `B` has been computed successfully on the GPU.
