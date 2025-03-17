#include <stdbool.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DEBUG_H
#define DEBUG_H

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line);

int verbose(const char * restrict, ...);
void conditionAssert(bool condition, std::string text, bool abort = false);
void setVerbose(bool);

#endif