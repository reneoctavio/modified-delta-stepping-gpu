/**
 * Modified Delta-Stepping Algorithm
 * Created on: Dec 2, 2015
 *
 * delta_stepping_sssp.cuh
 *
 * Author: Rene Octavio Queiroz Dias
 * License: GPLv3
 */

#ifndef DELTA_STEPPING_SSSP_H_
#define DELTA_STEPPING_SSSP_H_

#ifndef DEBUG_MSG
#define DEBUG_MSG 0
#endif

#ifndef DEBUG_MSG_BKT
#define DEBUG_MSG_BKT 0
#endif

#ifndef STATS
#define STATS 1
#endif

#ifndef STATS_WORK
#define STATS_WORK 0
#endif

// Includes
#include <sys/time.h>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/io/dimacs.h>
#include <cusp/io/matrix_market.h>

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "moderngpu.cuh"

using namespace mgpu;

// Typedefs
typedef typename cusp::csr_matrix<int, int, cusp::device_memory> DCsrMatrix;
typedef typename cusp::coo_matrix<int, int, cusp::device_memory> DCooMatrix;

typedef typename cusp::csr_matrix<int, int, cusp::host_memory> HCsrMatrix;
typedef typename cusp::coo_matrix<int, int, cusp::host_memory> HCooMatrix;

typedef typename cusp::array1d<int, cusp::device_memory> DVector;
typedef typename DVector::iterator DVectorIterator;

// Functions
void separate_graphs_device(DCsrMatrix *d_light, DCsrMatrix *d_heavy, DCsrMatrix *d_total_graph, const int k_delta);
void separate_graphs_host(HCsrMatrix *d_light, HCsrMatrix *d_heavy, HCsrMatrix *d_total_graph, const int k_delta);
void delta_stepping_gpu_mpgu(CudaContext& context, DCsrMatrix *d_graph, int ini_vertex, char* argv[]);

// Structs
struct is_heavy {
	const int k_upper_distance;
	is_heavy(int _upper) :
			k_upper_distance(_upper) {
	}

	__host__ __device__
	bool operator()(const int& x) {
		return x > k_upper_distance;
	}
};

struct is_light
{
	const int delta;
	is_light(int _delta) : delta(_delta) {}
	is_light() : delta(0) {}

	__host__ __device__
	bool operator()(const int& x)
	{
		return x <= delta;
	}
};

struct is_light_tuple
{
	const int delta;
	is_light_tuple(int _delta) : delta(_delta) {}
	is_light_tuple() : delta(0) {}

	template <typename Tuple>
	__host__ __device__
	bool operator()(Tuple tuple)
	{
		int edge_value = thrust::get<0>(tuple);
		return edge_value <= delta;
	}
};

struct get_valid_bucket
{
	const int bucket;
	get_valid_bucket(int _bucket) : bucket(_bucket) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple tuple)
	{
		int bucket_prev = thrust::get<0>(tuple);
		int bucket_dest = thrust::get<1>(tuple);
		int count_light = thrust::get<2>(tuple);
		int count_heavy = thrust::get<3>(tuple);

		if (bucket_prev == bucket) {
			thrust::get<1>(tuple) = INT_MAX;
			thrust::get<4>(tuple) = count_light;
			thrust::get<5>(tuple) = count_heavy;
		} else {
			thrust::get<1>(tuple) = bucket_prev;
			thrust::get<4>(tuple) = 0;
		}
	}
};

struct update_dist
{
	int* distance;
	int* graph_row;
	int* graph_column;
	int* graph_values;
	int* source_offsets;
	int* vertex_id_bkt;
	const int k_delta;

	update_dist(int* _dist, int* _row, int* _col, int* _val, int* _stl, int* _bkt, int _delta) :
		distance(_dist), graph_row(_row), graph_column(_col),
		graph_values(_val), source_offsets(_stl), vertex_id_bkt(_bkt),  k_delta(_delta) {}

	template <typename Tuple>
	__device__
	void operator()(Tuple tuple)
	{
		int origin_vx = thrust::get<0>(tuple);
		int origin_pos = thrust::get<1>(tuple);

		int target_index = graph_row[origin_vx] + origin_pos - source_offsets[origin_vx];

		int tent_distance = distance[origin_vx] + graph_values[target_index];
		int frontier_vertex = graph_column[target_index];

		if (tent_distance < atomicMin(&distance[frontier_vertex], tent_distance)) {
			int frontier_bucket = tent_distance / k_delta;
			atomicMin(&vertex_id_bkt[frontier_vertex], frontier_bucket);
		}
	}
};

#endif /* DELTA_STEPPING_SSSP_H_ */
