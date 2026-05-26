# PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME : NAVEEN KUMAR E</h3> 
<h3>ENTER YOUR REGISTER NO : 212224230181</h3> 
<h3>EX. NO : 06</h3> 
<h3>DATE : 26-05-2026</h3> 
<h1> <align=center> MATRIX TRANSPOSITION USING SHARED MEMORY </h3>
  Implement Matrix transposition using GPU Shared memory.</h3>

## AIM:
To perform Matrix Multiplication using Transposition using shared memory.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler

## PROCEDURE:
 CUDA_SharedMemory_AccessPatterns:

1. Begin Device Setup
    1.1 Select the device to be used for computation
    1.2 Retrieve the properties of the selected device
2. End Device Setup

3. Begin Array Size Setup
    3.1 Set the size of the array to be used in the computation
    3.2 The array size is determined by the block dimensions (BDIMX and BDIMY)
4. End Array Size Setup

5. Begin Execution Configuration
    5.1 Set up the execution configuration with a grid and block dimensions
    5.2 In this case, a single block grid is used
6. End Execution Configuration

7. Begin Memory Allocation
    7.1 Allocate device memory for the output array d_C
    7.2 Allocate a corresponding array gpuRef in the host memory
8. End Memory Allocation

9. Begin Kernel Execution
    9.1 Launch several kernel functions with different shared memory access patterns (Use any two patterns)
        9.1.1 setRowReadRow: Each thread writes to and reads from its row in shared memory
        9.1.2 setColReadCol: Each thread writes to and reads from its column in shared memory
        9.1.3 setColReadCol2: Similar to setColReadCol, but with transposed coordinates
        9.1.4 setRowReadCol: Each thread writes to its row and reads from its column in shared memory
        9.1.5 setRowReadColDyn: Similar to setRowReadCol, but with dynamic shared memory allocation
        9.1.6 setRowReadColPad: Similar to setRowReadCol, but with padding to avoid bank conflicts
        9.1.7 setRowReadColDynPad: Similar to setRowReadColPad, but with dynamic shared memory allocation
10. End Kernel Execution

11. Begin Memory Copy
    11.1 After each kernel execution, copy the output array from device memory to host memory
12. End Memory Copy

13. Begin Memory Free
    13.1 Free the device memory and host memory
14. End Memory Free

15. Reset the device

16. End of Algorithm

## PROGRAM:
```c
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter

cuda_code = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <math.h>

// Define matrix indexing for column-major order
#define index(i,j,ld) (((j)*(ld))+(i))

// Initialize matrices with smaller values for numerical stability
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[index(i, j, size)] = (float)(i + j) / size;
        }
    }
}

// CPU matrix multiplication (column-major order)
void cpuMatrixMultiplication(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[index(i, j, n)] = 0.0f;
            for (int k = 0; k < n; k++) {
                C[index(i, j, n)] += A[index(i, k, n)] * B[index(k, j, n)];
            }
        }
    }
}

int main() {
    int sizes[] = {256, 512, 1024};
    int numSizes = 3;

    for (int s = 0; s < numSizes; s++) {
        int size = sizes[s];
        printf("\\nRunning matrix multiplication for size: %d x %d\\n", size, size);

        // Allocate host memory (aligned to 32-byte boundaries)
        float *A = (float*)aligned_alloc(32, size * size * sizeof(float));
        float *B = (float*)aligned_alloc(32, size * size * sizeof(float));
        float *C_cpu = (float*)aligned_alloc(32, size * size * sizeof(float));
        float *C_gpu = (float*)aligned_alloc(32, size * size * sizeof(float));

        // Initialize matrices A and B
        initializeMatrix(A, size);
        initializeMatrix(B, size);

        // Timing CPU matrix multiplication
        clock_t start_cpu = clock();
        cpuMatrixMultiplication(A, B, C_cpu, size);
        clock_t end_cpu = clock();

        double time_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;

        printf("CPU Matrix Multiplication Time: %f seconds\\n", time_cpu);

        float *d_A, *d_B, *d_C;

        cudaMalloc((void**)&d_A, size * size * sizeof(float));
        cudaMalloc((void**)&d_B, size * size * sizeof(float));
        cudaMalloc((void**)&d_C, size * size * sizeof(float));

        // Copy matrices from host to device
        cudaMemcpy(d_A, A, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size * size * sizeof(float), cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Matrix multiplication using cuBLAS (column-major order)
        cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    size,
                    size,
                    size,
                    &alpha,
                    d_B,
                    size,
                    d_A,
                    size,
                    &beta,
                    d_C,
                    size);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_gpu;

        cudaEventElapsedTime(&time_gpu, start, stop);

        printf("GPU Matrix Multiplication Time (cuBLAS): %f milliseconds\\n", time_gpu);

        // Copy result back to host
        cudaMemcpy(C_gpu, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify the results using relative error
        int errors = 0;
        float max_relative_error = 1e-4;

        for (int i = 0; i < size * size; i++) {

            float relative_error =
                fabs(C_cpu[i] - C_gpu[i]) /
                fmax(fabs(C_cpu[i]), fabs(C_gpu[i]));

            if (relative_error > max_relative_error) {
                errors++;
            }
        }

        if (errors == 0) {
            printf("Results verified successfully for size %d x %d\\n", size, size);
        } else {
            printf("Discrepancies found in the results for size %d x %d\\n", size, size);
        }

        // Clean up
        cublasDestroy(handle);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu);
    }

    return 0;
}
"""

# Save the CUDA code to a file
with open("matrix_multiplication.cu", "w") as file:
    file.write(cuda_code)
```

## OUTPUT:
<img width="1813" height="425" alt="image" src="https://github.com/user-attachments/assets/7b2cde8c-b590-44e8-a951-1f2cb66f622a" />


## RESULT:
Thus the program has been executed by using CUDA to transpose a matrix. It is observed that there are variations shared memory and global memory implementation. The elapsed times are recorded as _______________.
