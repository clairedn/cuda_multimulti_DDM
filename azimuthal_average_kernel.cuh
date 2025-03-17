//////////////////////////////////////
//  Reduction code is based heavily on the reduction_example from Nvidia's CUDA SDK examples
//  See "Optimizing parallel reduction in CUDA" - M. Harris for more details
//  some tweaks in regard to adding Boolean mask made
//////////////////////////////////////

#include <stdio.h>
#include <cuda_runtime.h>

#ifndef AZIMUTH_KERNEL
#define AZIMUTH_KERNEL

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// Specialise for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};


bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }


template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		mySum += __shfl_down_sync(mask, mySum, offset);
	}
	return mySum;
}


#if __CUDA_ARCH__ >= 800
// Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
// when on SM 8.0 or higher
template <>
__device__ __forceinline__ int warpReduceSum<int>(unsigned int mask,
                                                  int mySum) {
  mySum = __reduce_add_sync(mask, mySum);
  return mySum;
}
#endif


template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void kernelMaskReduce(const T* __restrict__ d_idata,
						const bool* __restrict d_mask,
						T* __restrict__ d_odata,
						unsigned int n) {

	T *sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int gridSize = blockSize * gridDim.x;
	unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
	maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
	const unsigned int mask = (0xffffffff) >> maskLength;

	T mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread

	if (nIsPow2) {
		unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
		gridSize = gridSize << 1;

		while (i < n) {
			if (d_mask[i]) { mySum += d_idata[i]; }

			// ensure we don't read out of bounds -- this is optimized away for
			// powerOf2 sized arrays
			if ((i + blockSize) < n) {
				if (d_mask[i + blockSize]) { mySum += d_idata[i + blockSize]; }
			}
			i += gridSize;
		}
	} else {
	    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
	    while (i < n) {
			if (d_mask[i]) { mySum += d_idata[i]; }
			i += gridSize;
	    }
	}

	// Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
	// SM 8.0
	mySum = warpReduceSum<T>(mask, mySum);

  // each thread puts its local sum into shared memory
	if ((tid % warpSize) == 0) {
		sdata[tid / warpSize] = mySum;
	}

	__syncthreads();

	const unsigned int shmem_extent =
			(blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
	const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
	if (tid < shmem_extent) {
		mySum = sdata[tid];
		// Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
		// SM 8.0
		mySum = warpReduceSum<T>(ballot_result, mySum);
	}

	// write result for this block to global mem
	if (tid == 0) { d_odata[blockIdx.x] = mySum; }
}


template <class T>
void maskReduce(int size, int threads, int blocks, T *d_idata, bool *d_mask, T *d_odata) {
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// choose which of the optimized versions of reduction to launch
	// For reduce kernel we require only blockSize/warpSize
	// number of elements in shared memory
	int smemSize = ((threads / 32) + 1) * sizeof(T);
	if (isPow2(size)) {
		switch (threads) {
			case 512:
				kernelMaskReduce<T, 512, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 256:
				kernelMaskReduce<T, 256, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 128:
				kernelMaskReduce<T, 128, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 64:
				kernelMaskReduce<T, 64, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 32:
				kernelMaskReduce<T, 32, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 16:
				kernelMaskReduce<T, 16, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 8:
				kernelMaskReduce<T, 8, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 4:
				kernelMaskReduce<T, 4, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 2:
				kernelMaskReduce<T, 2, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 1:
				kernelMaskReduce<T, 1, true>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
		}
	} else {
		switch (threads) {
			case 512:
				kernelMaskReduce<T, 512, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 256:
				kernelMaskReduce<T, 256, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 128:
				kernelMaskReduce<T, 128, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 64:
				kernelMaskReduce<T, 64, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 32:
				kernelMaskReduce<T, 32, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 16:
				kernelMaskReduce<T, 16, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 8:
				kernelMaskReduce<T, 8, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 4:
				kernelMaskReduce<T, 4, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 2:
				kernelMaskReduce<T, 2, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
			case 1:
				kernelMaskReduce<T, 1, false>
				<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_mask, d_odata, size);
				break;
		}
	}
}

template void maskReduce<float>(int size, int threads, int blocks, float *d_idata, bool *d_mask, float *d_odata);
template void maskReduce<double>(int size, int threads, int blocks, double *d_idata, bool *d_mask, double *d_odata);

#endif