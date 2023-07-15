 #include <stdio.h> 
 #include <assert.h> 
 #include <cuda.h> 
 #include <cuda_runtime.h>

 #define N 2

 // Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
  float A[N][N] = {1.0, 1.0, 1.0, 1.0};
  float B[N][N] = {1.0, 1.0, 1.0, 1.0};
  float C[N][N] = {0.0, 0.0, 0.0, 0.0};

  float(*d_A)[N], (*d_B)[N], (*d_C)[N];

  // Allocate device memory
  cudaMalloc((void**)&d_A, sizeof(float)*N*N);
  cudaMalloc((void**)&d_B, sizeof(float)*N*N);
  cudaMalloc((void**)&d_C, sizeof(float)*N*N);

  // Copy host memory to device
  cudaMemcpy(d_A, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  // Kernel invocation
  // dim3 threadsPerBlock(16, 16);
  // dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
  dim3 threadsPerBlock(N, N); // 2x2 for a 2x2 matrix 
  dim3 numBlocks(1, 1); // Only one block needed 
  MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

  // Copy result from device memory to host
  cudaMemcpy(C, d_C, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

  // Print result
  for(int i = 0; i < N; i++) {
      for(int j = 0; j < N; j++) {
          printf("%f ", C[i][j]);
      }
      printf("\n");
  }

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

return 0;
}
