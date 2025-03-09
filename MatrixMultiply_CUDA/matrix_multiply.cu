#include <iostream>
#include <cuda_runtime.h>
// following code illustrates the gpu kernel for matrix multiplication...
__global__ void matrixMultiply(int *A, int *B, int *C, int M, int N, int P)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index of C

    if (row < M && col < P) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

// function ,, to launch matrix multiplication kernel
void launchMatrixMultiply(int *A, int *B, int *C, int M, int N, int P)
{
    int *d_A, *d_B, *d_C;

    // allocate memory on the GPU
    cudaMalloc(&d_A, M * N * sizeof(int));
    cudaMalloc(&d_B, N * P * sizeof(int));
    cudaMalloc(&d_C, M * P * sizeof(int));

    // copy data from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * P * sizeof(int), cudaMemcpyHostToDevice);

    // define block and grid size
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the matrix multiply kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, P);

    // Check for errors in kernel launch
    cudaDeviceSynchronize();

    // Copy the result matrix back to the host
    cudaMemcpy(C, d_C, M * P * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    // Define the dimensions of the matrices
    int M = 3;  // Rows in A and C
    int N = 3;  // Columns in A and Rows in B
    int P = 3;  // Columns in B and C

    // allocate memory for matrices A, B, and C
    int A[M * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[N * P] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int C[M * P]; // Result matrix

    // launch matrix multiplication on GPU
    launchMatrixMultiply(A, B, C, M, N, P);

    // / print the result
    std::cout << "Matrix C (Result):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            std::cout << C[i * P + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
