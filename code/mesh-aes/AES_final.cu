// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <ctime>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <device_launch_parameters.h>
//#include <device_functions.h>

// Custom header 
#include "AES_final.h"
//
#include "128-cmac.cuh"

int main() {
	cudaSetDevice(0);
	CMAC128ExhaustiveSearch();
	return 0;
}
