
#include "../include/utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(func)                                                     	   \
	do {                                                                           \
		cudaError_t status = (func);                                               \
		if (status != cudaSuccess) {                                               \
			printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,   \
				cudaGetErrorString(status), status);                               \
			exit(EXIT_FAILURE);                                                    \
		}                                                                          \
	} while (0)

#define CHECK(name)                                                          \
	float *d_Aref_##name, *d_Bref_##name, *d_Cref_##name;                   \
	std::cerr << "checking " << #name << std::endl;                         \
	CUDA_CHECK(cudaMalloc(&d_Aref_##name, Ref::M * Ref::K * sizeof(float)));\
	CUDA_CHECK(cudaMalloc(&d_Bref_##name, Ref::K * Ref::N * sizeof(float)));\
	CUDA_CHECK(cudaMalloc(&d_Cref_##name, Ref::M * Ref::N * sizeof(float)));\
	CUDA_CHECK(cudaMemcpy(d_Aref_##name, ref.A, Ref::M * Ref::K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_Bref_##name, ref.B, Ref::K * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	for (int i_chk = 0; i_chk < Ref::M * Ref::N; i_chk++) {                 \
		refC[i_chk] = 0.0f;                                                \
	}                                                                       \
	CUDA_CHECK(cudaMemcpy(d_Cref_##name, refC, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	name(d_Aref_##name, d_Bref_##name, d_Cref_##name, Ref::M, Ref::N, Ref::K); \
	{                                                                       \
		cudaError_t err_c_##name = cudaGetLastError();                      \
		if (err_c_##name != cudaSuccess) {                                  \
			std::cerr << "CUDA Error: " << cudaGetErrorString(err_c_##name) << std::endl; \
		}                                                                   \
	}                                                                       \
	CUDA_CHECK(cudaMemcpy(refC, d_Cref_##name, Ref::M * Ref::N * sizeof(float), cudaMemcpyDeviceToHost)); \
	if (!ref.checkRef(refC)) {                                              \
		std::cerr << #name << ": check ref failed!" << std::endl;           \
	}                                                                       \
	cudaFree(d_Aref_##name);                                               \
	cudaFree(d_Bref_##name);                                               \
	cudaFree(d_Cref_##name);

#define TIME(name)                                                                       \
	float *d_A_##name, *d_B_##name, *d_C_##name;                                         \
	CUDA_CHECK(cudaMalloc(&d_A_##name, M * K * sizeof(float)));                          \
	CUDA_CHECK(cudaMalloc(&d_B_##name, K * N * sizeof(float)));                          \
	CUDA_CHECK(cudaMalloc(&d_C_##name, M * N * sizeof(float)));                          \
	CUDA_CHECK(cudaMemcpy(d_A_##name, A, M * K * sizeof(float), cudaMemcpyHostToDevice));\
	CUDA_CHECK(cudaMemcpy(d_B_##name, B, K * N * sizeof(float), cudaMemcpyHostToDevice));\
	cudaEvent_t start_##name, end_##name;                                                \
	cudaEventCreate(&start_##name);                                                      \
	cudaEventCreate(&end_##name);                                                        \
	/* Warm up */                                                                        \
	for (int i_wu = 0; i_wu < 2; i_wu++)                                                 \
	{                                                                                    \
		CUDA_CHECK(cudaMemset(d_C_##name, 0, M * N * sizeof(float)));                    \
		name(d_A_##name, d_B_##name, d_C_##name, M, N, K);                               \
	}                                                                                    \
	float milliseconds_##name = 0.0f;                                                    \
	/* Timed runs */                                                                     \
	for (int it = 0; it < 3; it++)                                                       \
	{                                                                                    \
		CUDA_CHECK(cudaMemset(d_C_##name, 0, M * N * sizeof(float)));                    \
		cudaDeviceSynchronize();                                                         \
		cudaEventRecord(start_##name);                                                   \
		name(d_A_##name, d_B_##name, d_C_##name, M, N, K);                               \
		cudaEventRecord(end_##name);                                                     \
		cudaEventSynchronize(end_##name);                                                \
		float thisTime = 0.0f;                                                           \
		cudaEventElapsedTime(&thisTime, start_##name, end_##name);                       \
		milliseconds_##name += thisTime;                                                 \
	}                                                                                    \
	std::cout << "Time taken for GEMM (GPU, " << #name << "): "                          \
	          << milliseconds_##name << " ms (total over 3 runs)" << std::endl;          \
	cudaFree(d_A_##name);                                                                \
	cudaFree(d_B_##name);                                                                \
	cudaFree(d_C_##name);                                                                \
	cudaEventDestroy(start_##name);                                                      \
	cudaEventDestroy(end_##name);


__global__ void gemm_gpu_o0_kernel(float* A, float* B, float *C,
                                   int M, int N, int K)
{
	// Only block (0,0) and thread(0,0) does the entire job
	if (threadIdx.x == 0 && blockIdx.x == 0 &&
	    threadIdx.y == 0 && blockIdx.y == 0)
	{
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				float sum = 0.0f;
				for (int kk = 0; kk < K; kk++) {
					sum += A[i*K + kk] * B[kk*N + j];
				}
				C[i*N + j] = sum;
			}
		}
    }
}

void gemm_gpu_o0(float* A, float* B, float* C,
                 int M, int N, int K)
{
	dim3 blockSize(1, 1);
	dim3 gridSize(1, 1);
	gemm_gpu_o0_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}


__global__ void gemm_gpu_o1_kernel(float* A, float* B, float *C,
                                   int M, int N, int K)
{
	// Each thread computes one (row,col) of the output
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N) {
		float sum = 0.0f;
		for (int kk = 0; kk < K; kk++) {
			sum += A[row * K + kk] * B[kk * N + col];
		}
		C[row * N + col] = sum;
	}
}

void gemm_gpu_o1(float* A, float* B, float* C,
                 int M, int N, int K)
{
	// Typically 16x16 or 32x32 is a good start
	dim3 block(16, 16);
	dim3 grid( (N + block.x - 1)/block.x,
	           (M + block.y - 1)/block.y );
	gemm_gpu_o1_kernel<<<grid, block>>>(A, B, C, M, N, K);
}


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

__global__ void gemm_gpu_o2_kernel(float* A, float* B, float *C,
                                   int M, int N, int K)
{
	// Tiling in shared memory
	__shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	float sum = 0.0f;
	// Loop over tiles of size BLOCK_SIZE in K dimension
	for (int t = 0; t < ( (K + BLOCK_SIZE - 1) / BLOCK_SIZE ); t++)
	{
		int tiledK = t * BLOCK_SIZE; // start index of this tile in K

		// Load A tile into shared memory
		if ( (row < M) && (tiledK + threadIdx.x < K) ) {
			sA[threadIdx.y][threadIdx.x] = A[row*K + (tiledK + threadIdx.x)];
		} else {
			sA[threadIdx.y][threadIdx.x] = 0.0f;
		}

		// Load B tile into shared memory
		if ( (col < N) && (tiledK + threadIdx.y < K) ) {
			sB[threadIdx.y][threadIdx.x] = B[(tiledK + threadIdx.y)*N + col];
		} else {
			sB[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		// Compute partial sums
		for (int kk = 0; kk < BLOCK_SIZE; kk++) {
			sum += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
		}

		__syncthreads();
	}

	// Write back result
	if (row < M && col < N) {
		C[row*N + col] = sum;
	}
}

void gemm_gpu_o2(float* A, float* B, float* C,
                 int M, int N, int K)
{
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid( (N + BLOCK_SIZE - 1)/BLOCK_SIZE,
	           (M + BLOCK_SIZE - 1)/BLOCK_SIZE );
	gemm_gpu_o2_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

#ifndef BLOCK_SIZE_O3
#define BLOCK_SIZE_O3 32
#endif

__global__ void gemm_gpu_o3_kernel(float* A, float* B, float *C,
                                   int M, int N, int K)
{
	// Same tiling approach, just with a different tile size
	__shared__ float sA[BLOCK_SIZE_O3][BLOCK_SIZE_O3];
	__shared__ float sB[BLOCK_SIZE_O3][BLOCK_SIZE_O3];

	int row = blockIdx.y * BLOCK_SIZE_O3 + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE_O3 + threadIdx.x;

	float sum = 0.0f;
	for (int t = 0; t < ( (K + BLOCK_SIZE_O3 - 1) / BLOCK_SIZE_O3 ); t++)
	{
		int tiledK = t * BLOCK_SIZE_O3;

		// Load A
		if ( (row < M) && (tiledK + threadIdx.x < K) ) {
			sA[threadIdx.y][threadIdx.x] = A[row*K + (tiledK + threadIdx.x)];
		} else {
			sA[threadIdx.y][threadIdx.x] = 0.0f;
		}

		// Load B
		if ( (col < N) && (tiledK + threadIdx.y < K) ) {
			sB[threadIdx.y][threadIdx.x] = B[(tiledK + threadIdx.y)*N + col];
		} else {
			sB[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		// Partial sum
		for (int kk = 0; kk < BLOCK_SIZE_O3; kk++) {
			sum += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
		}
		__syncthreads();
	}

	if (row < M && col < N) {
		C[row*N + col] = sum;
	}
}

void gemm_gpu_o3(float* A, float* B, float* C,
                 int M, int N, int K)
{
	dim3 block(BLOCK_SIZE_O3, BLOCK_SIZE_O3);
	dim3 grid( (N + BLOCK_SIZE_O3 - 1)/BLOCK_SIZE_O3,
	           (M + BLOCK_SIZE_O3 - 1)/BLOCK_SIZE_O3 );
	gemm_gpu_o3_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char* argv[])
{
	if (argc < 4) {
		std::cout << "Usage: " << argv[0] << " <M> <N> <K>\n";
		return 1;
	}

	int M = std::atoi(argv[1]);
	int N = std::atoi(argv[2]);
	int K = std::atoi(argv[3]);

	std::cout << "Running GEMM on GPU with sizes: M=" << M
	          << ", N=" << N << ", K=" << K << std::endl;

	// Allocate host buffers
	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	// Fill with random data
	fillRandom(A, M * K);
	fillRandom(B, K * N);

	// For correctness check against CPU reference
	auto ref = Ref();
	float* refC = new float[Ref::M * Ref::N]();
	for (int i = 0; i < Ref::M * Ref::N; i++)
		refC[i] = 0.0f; // ensure zero


	CHECK(gemm_gpu_o0)
	CHECK(gemm_gpu_o1)
	CHECK(gemm_gpu_o2)
	CHECK(gemm_gpu_o3)


	TIME(gemm_gpu_o0)
	TIME(gemm_gpu_o1)
	TIME(gemm_gpu_o2)
	TIME(gemm_gpu_o3)

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] refC;

	return 0;
}
