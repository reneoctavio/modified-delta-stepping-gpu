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
#define DEBUG_MSG_BKT 1
#endif

#ifndef STATS
#define STATS 0
#endif

#ifndef STATS_WORK
#define STATS_WORK 0
#endif

// Includes
#include <sys/time.h>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/io/dimacs.h>
#include <cusp/io/matrix_market.h>

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/pair.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
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

typedef typename thrust::tuple<DVectorIterator, DVectorIterator> Tuple2DVec;
typedef typename thrust::zip_iterator<Tuple2DVec> Zip2DVecIterator;

typedef typename thrust::tuple<DVectorIterator, DVectorIterator, DVectorIterator> Tuple3DVec;
typedef typename thrust::zip_iterator<Tuple3DVec> Zip3DVecIterator;

// Functions
void delta_stepping_gpu_sssp(DCsrMatrix *d_graph_light, DCsrMatrix *d_graph_heavy, DVector *d_distance, const int k_delta, int ini_vertex);

void expand_edges(DCsrMatrix *d_graph, DVector *d_distance,
		DVector *d_count, DVector *d_origin, int num_origin_vertices,
		DVector *d_frontier_vertices, DVector *d_tentative_distance);

void remove_invalid_distances(DVector *d_distance, DVector *d_frontier_vertices, DVector *d_tent_distance);

void relax(DVector *d_distance, DVector *vertex_id_bucket, DVector *d_frontier_vertices, DVector *d_tent_distance, const int k_delta);

void separate_graphs(DCsrMatrix *d_light, DCsrMatrix *d_heavy, DCsrMatrix *d_total_graph, const int k_delta);
void separate_graphs_host(HCsrMatrix *d_light, HCsrMatrix *d_heavy, HCsrMatrix *d_total_graph, const int k_delta);

void delta_stepping_gpu_mpgu(CudaContext& context, HCsrMatrix *h_graph_light,
		HCsrMatrix *h_graph_heavy, const int k_delta, int ini_vertex);
// Structs
struct is_inside_bucket
{
	const int bucket;
	is_inside_bucket(int _bucket) : bucket(_bucket) {}

	__host__ __device__
	int operator()(const int& bucket_num)
	{
		return (bucket_num == bucket);
	}
};

struct update_distance_and_vertices
{
	int* distance;
	int* frontier_vertices;
	int* graph_target_vertices;
	int* graph_values;
	update_distance_and_vertices(int *_dist, int *_front, int *_tgt, int* _val) :
		distance(_dist), frontier_vertices(_front), graph_target_vertices(_tgt), graph_values(_val) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple tuple)
	{
		int graph_begin_offset = thrust::get<0>(tuple);
		int orgin_vertex_distance = thrust::get<1>(tuple);
		int target_begin_offset = thrust::get<2>(tuple);
		int target_end_offset = thrust::get<3>(tuple);
		for (int i = target_begin_offset; i < target_end_offset; i++) {
			int graph_offset = graph_begin_offset + (i - target_begin_offset);
			distance[i] = orgin_vertex_distance + graph_values[graph_offset];
			frontier_vertices[i] = graph_target_vertices[graph_offset];
		}
	}
};

struct update_distance
{
	const int k_delta;
	update_distance(int _delta) : k_delta(_delta) {}

	template <typename Tuple>
	__device__
	void operator()(Tuple tuple)
	{
		int tent_distance = thrust::get<0>(tuple);
		int new_bucket = tent_distance / k_delta;

		int &tuple1 = thrust::get<1>(tuple);
		int *distance = const_cast<int*>(&tuple1);
		atomicMin(distance, tent_distance);

		int &tuple2 = thrust::get<2>(tuple);
		int *vx_id_bucket = const_cast<int*>(&tuple2);
		atomicMin(vx_id_bucket, new_bucket);
	}
};

struct is_invalid_tent_distance {
	int* distance_begin;
	is_invalid_tent_distance(int* _distance_begin) : distance_begin(_distance_begin) {}

	template<typename Tuple>
	__host__ __device__
	bool operator()(Tuple tuple) {
		int cur_dist = distance_begin[thrust::get<0>(tuple)];
		return thrust::get<1>(tuple) >= cur_dist;
	}
};

struct copy_non_deleted
{
	DVectorIterator deleted_iter;
	copy_non_deleted(DVectorIterator _deleted_iter) : deleted_iter(_deleted_iter) {}
	copy_non_deleted() : deleted_iter(DVectorIterator()) {}

	__host__ __device__
	bool operator()(const int& vertex) const
	{
		return ! *(deleted_iter + vertex);
	}
};

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

struct is_valid_bucket
{
	__host__ __device__
	bool operator()(const int& bucket)
	{
		return bucket != INT_MAX;
	}
};


void relax_csr(DCsrMatrix *d_graph, DVector *d_distance,
		DVector *d_vertex_id_bucket, DVector *d_deleted,
		const int k_delta, const int k_cur_bucket);

void delta_stepping_gpu_csr(DCsrMatrix *d_graph_light, DCsrMatrix *d_graph_heavy,
		DVector *d_distance, const int k_delta, int ini_vertex);

struct update_csr
{
	int* distance;
	int* graph_column;
	int* graph_values;
	int* vertex_id_bucket;
	const int k_delta;
	const int k_cur_bucket;

	update_csr(int* _dist, int* _col, int* _val, int* _stl, int _delta, int _bkt) :
		distance(_dist), graph_column(_col), graph_values(_val),
		vertex_id_bucket(_stl), k_delta(_delta), k_cur_bucket(_bkt) {}

	template <typename Tuple>
	__device__
	void operator()(Tuple tuple)
	{
		// Check if vertex was deleted
		int deleted = thrust::get<3>(tuple);
		if (!deleted) return;

		thrust::get<3>(tuple) = 0;

		// Get offsets
		int offset_begin = thrust::get<0>(tuple);
		int offset_end = thrust::get<1>(tuple);

		// Get origin distance
		int origin_distance = thrust::get<2>(tuple);

		for (int i = offset_begin; i < offset_end; i++) {
			int tent_distance = origin_distance + graph_values[i];
			int frontier_bucket = tent_distance / k_delta;
			int frontier_vertex = graph_column[i];

			int old = atomicMin(&distance[frontier_vertex], tent_distance);

			if (tent_distance < old) atomicMin(&vertex_id_bucket[frontier_vertex], frontier_bucket);
		}
	}
};

struct remove_origin
{
	template <typename Tuple>
	__host__ __device__
	int operator()(Tuple tuple)
	{
		return thrust::get<1>(tuple) ? INT_MAX : thrust::get<0>(tuple);
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
