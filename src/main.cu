/**
 * Modified Delta-Stepping Algorithm
 * Created on: Nov 13, 2015
 *
 * main.cu
 *
 * Author: Rene Octavio Queiroz Dias
 * License: GPLv3
 */

#include "delta_stepping_sssp.h"

int main(int argc, char* argv[]) {
	// Check correct number of arguments

#if !STATS
	// 1 - Type; 2 - Graph File; 3 - Delta
	if (argc != 4) return 1;

#else
	// 1 - Type; 2 - Graph File; 3 - Delta Array File; 4 - Ground Truth File; 5 - Number of Samples
	if (argc != 6) return 1;
#endif

	ContextPtr context = CreateCudaDevice(0);

	DCsrMatrix d_graph;
	int delta = std::atoi(argv[3]);

	if 		(std::strcmp(argv[1], "d") == 0) { cusp::io::read_dimacs_file(d_graph, argv[2]); }
	else if (std::strcmp(argv[1], "m") == 0) { cusp::io::read_matrix_market_file(d_graph, argv[2]); }
	else 					 				 { printf("Invalid format.\n"); return 1; }

	delta_stepping_gpu_mpgu(*context, &d_graph, 0, argv);

	return 0;
}
