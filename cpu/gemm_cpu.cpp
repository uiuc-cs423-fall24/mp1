#include <chrono>
#include "../include/utils.h"

#define CHECK(name) \
  std::cout << "checking " << #name << std::endl;		\
  for (int j = 0; j < Ref::N; j++) {		\
    for (int i = 0; i < Ref::M; i++) {    		\
      refC[i * Ref::N + j] = 0;		\
	}		\
  }		\
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);		\
  if (!ref.checkRef(refC)){					\
    std::cerr << #name << ": check ref failed!" << std::endl;	\
  };								\
  fillRandom(refC, Ref::M * Ref::N);				
  
#define TIME(name) \
  for (int i = 0; i < 2; i++)						\
    {									\
	  for (int j = 0; j < N; j++) {		\
	    for (int k = 0; k < M; k++) {    		\
	      C[k * N + j] = 0;		\
	    }		\
	  }		\
      name(A, B, C, M, N, K);						\
    }									\
  double time_ ## name = 0.0; \
  for (int i = 0; i < 3; i++)						\
    {									\
	  for (int j = 0; j < N; j++) {		\
	    for (int k = 0; k < M; k++) {    		\
	      C[k * N + j] = 0;		\
	    }		\
	  }		\
	  auto start_time_ ## name = std::chrono::high_resolution_clock::now(); \
      name(A, B, C, M, N, K);						\
	  auto end_time_ ## name = std::chrono::high_resolution_clock::now();	\
  	  std::chrono::duration<double, std::milli> duration_ ## name = end_time_ ## name - start_time_ ## name; \
	  time_ ## name += duration_ ## name.count(); \
    }									\
  std::cout << "Time taken for GEMM (CPU," << #name <<"): " << time_ ## name << "ms" << std::endl; 


// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
void gemm_cpu_o0(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
	C[i * N + j]  += A[i * K + k]  * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler flags. So, you can use o3's code for that part
void gemm_cpu_o1(float* A, float* B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      float a_val = A[i * K + k];
      for (int j = 0; j < N; j++) {
        C[i * N + j] += a_val * B[k * N + j];
      }
    }
  }
}




void gemm_cpu_o2(float* A, float* B, float* C, int M, int N, int K) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 64;

  for (int i0 = 0; i0 < M; i0 += BM) {
    int iMax = std::min(i0 + BM, M);
    for (int k0 = 0; k0 < K; k0 += BK) {
      int kMax = std::min(k0 + BK, K);
      for (int j0 = 0; j0 < N; j0 += BN) {
        int jMax = std::min(j0 + BN, N);

        for (int i = i0; i < iMax; i++) {
          for (int k = k0; k < kMax; k++) {
            float aVal = A[i*K + k];
            for (int j = j0; j < jMax; j++) {
              C[i*N + j] += aVal * B[k*N + j];
            }
          }
        }
      }
    }
  }
}



#include <omp.h>



void gemm_cpu_o3(float* A, float* B, float* C, int M, int N, int K) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 64;

  #pragma omp parallel for shared(A, B, C, M, N, K)
  for (int i0 = 0; i0 < M; i0 += BM) {
    int iMax = std::min(i0 + BM, M);
    for (int k0 = 0; k0 < K; k0 += BK) {
      int kMax = std::min(k0 + BK, K);
      for (int j0 = 0; j0 < N; j0 += BN) {
        int jMax = std::min(j0 + BN, N);
        for (int i = i0; i < iMax; i++) {
          for (int k = k0; k < kMax; k++) {
            float aVal = A[i*K + k];
            #pragma omp simd
            for (int j = j0; j < jMax; j++) {
              C[i*N + j] += aVal * B[k*N + j];
            }
          }
        }
      }
    }
  }
}






int main(int argc, char* argv[]) {
	if (argc < 3) {
	  std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
	  return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);


	float* refC = new float[Ref::M * Ref::N]();
	auto ref = Ref();
	CHECK(gemm_cpu_o0)
	CHECK(gemm_cpu_o1)
	CHECK(gemm_cpu_o2)
	CHECK(gemm_cpu_o3)
	delete[] refC;
	
	TIME(gemm_cpu_o0)
	TIME(gemm_cpu_o1)
	TIME(gemm_cpu_o2)
	TIME(gemm_cpu_o3)

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}
