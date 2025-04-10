#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>

#include <string>

#include "debug.hpp"

void gpuAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "[GPU Assert] %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(EXIT_FAILURE);
	}
}

bool Verbose = false;
void setVerbose(bool setting) { Verbose = setting; }

int verbose( const char * format, ...) {
	if ( !Verbose ) {
		return 0;
	}

	va_list args;
	va_start(args, format);
	int ret = vprintf(format, args);
	va_end(args);

	return ret;
}

void conditionAssert(bool condition, std::string text, bool abort) {
    if (condition == false) {
        if (abort) {
            std::string out = "[Error] " + text + "\n";
            fprintf(stderr, "%s", out.c_str());
            exit(EXIT_FAILURE);
        } else {
            std::string out = "[Warning] " + text + "\n";
            fprintf(stderr, "%s", out.c_str());
        }

    }
}
