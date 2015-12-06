/**
 * Modified Delta-Stepping Algorithm
 * Created on: Nov 13, 2015
 *
 * delta_stepping_sssp.cu
 *
 * Author: Rene Octavio Queiroz Dias
 * License: GPLv3
 */

#include "delta_stepping_sssp.h"

/**
 * Separate light and heavy graphs based on the original graph
 * Edges are light if less and equal to delta, and heavy otherwise
 * @param d_light a light CSR graph (out)
 * @param d_heavy a heavy CSR graph (out)
 * @param d_total_graph the complete CSR graph (in)
 * @param k_delta a constant delta
 */
void separate_graphs_host(HCsrMatrix *d_light, HCsrMatrix *d_heavy, HCsrMatrix *d_total_graph, const int k_delta)
{
	// COO Graphs
	HCooMatrix d_graph, d_graph_light, d_graph_heavy;

	// Convert CSR to COO
	cusp::convert(*d_total_graph, d_graph);

	// Update quantity of rows, columns and edges; also allocate memory for separation
	int qty_vertices = d_graph.num_rows;

	int qty_light_edges = thrust::count_if(d_graph.values.begin(), d_graph.values.end(), is_light(k_delta));
	int qty_heavy_edges = thrust::count_if(d_graph.values.begin(), d_graph.values.end(), is_heavy(k_delta));

	d_graph_light.num_cols = qty_vertices;
	d_graph_heavy.num_cols = qty_vertices;
	d_graph_light.num_rows = qty_vertices;
	d_graph_heavy.num_rows = qty_vertices;

	d_graph_light.num_entries = qty_light_edges;
	d_graph_heavy.num_entries = qty_heavy_edges;

	d_graph_light.values.resize(qty_light_edges);
	d_graph_light.values.shrink_to_fit();
	d_graph_heavy.values.resize(qty_heavy_edges);
	d_graph_heavy.values.shrink_to_fit();

	d_graph_light.column_indices.resize(qty_light_edges);
	d_graph_light.column_indices.shrink_to_fit();
	d_graph_heavy.column_indices.resize(qty_heavy_edges);
	d_graph_heavy.column_indices.shrink_to_fit();

	d_graph_light.row_indices.resize(qty_light_edges);
	d_graph_light.row_indices.shrink_to_fit();
	d_graph_heavy.row_indices.resize(qty_heavy_edges);
	d_graph_heavy.row_indices.shrink_to_fit();

	// Partition
	thrust::stable_partition_copy(thrust::host,
			thrust::make_zip_iterator(thrust::make_tuple(
					d_graph.values.begin(),
					d_graph.column_indices.begin(),
					d_graph.row_indices.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(
					d_graph.values.end(),
					d_graph.column_indices.end(),
					d_graph.row_indices.end())),
			thrust::make_zip_iterator(thrust::make_tuple(
					d_graph_light.values.begin(),
					d_graph_light.column_indices.begin(),
					d_graph_light.row_indices.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(
					d_graph_heavy.values.begin(),
					d_graph_heavy.column_indices.begin(),
					d_graph_heavy.row_indices.begin())),
			is_light_tuple(k_delta));

	// Convert COO to CSR
	cusp::convert(d_graph_light, *d_light);
	cusp::convert(d_graph_heavy, *d_heavy);

	d_light->row_offsets.shrink_to_fit();
	d_light->column_indices.shrink_to_fit();
	d_light->values.shrink_to_fit();

	d_heavy->row_offsets.shrink_to_fit();
	d_heavy->column_indices.shrink_to_fit();
	d_heavy->values.shrink_to_fit();
}

/**
 * Separate light and heavy graphs based on the original graph
 * Edges are light if less and equal to delta, and heavy otherwise
 * @param d_light a light CSR graph (out)
 * @param d_heavy a heavy CSR graph (out)
 * @param d_total_graph the complete CSR graph (in)
 * @param k_delta a constant delta
 */
void separate_graphs_device(DCsrMatrix *d_light, DCsrMatrix *d_heavy, DCsrMatrix *d_total_graph, const int k_delta)
{
	// COO Graphs
	DCooMatrix d_graph, d_graph_light, d_graph_heavy;

	// Convert CSR to COO
	cusp::convert(*d_total_graph, d_graph);

	// Update quantity of rows, columns and edges; also allocate memory for separation
	int qty_vertices = d_graph.num_rows;

	int qty_light_edges = thrust::count_if(d_graph.values.begin(), d_graph.values.end(), is_light(k_delta));
	int qty_heavy_edges = thrust::count_if(d_graph.values.begin(), d_graph.values.end(), is_heavy(k_delta));

	d_graph_light.num_cols = qty_vertices;
	d_graph_heavy.num_cols = qty_vertices;
	d_graph_light.num_rows = qty_vertices;
	d_graph_heavy.num_rows = qty_vertices;

	d_graph_light.num_entries = qty_light_edges;
	d_graph_heavy.num_entries = qty_heavy_edges;

	d_graph_light.values.resize(qty_light_edges);
	d_graph_light.values.shrink_to_fit();
	d_graph_heavy.values.resize(qty_heavy_edges);
	d_graph_heavy.values.shrink_to_fit();

	d_graph_light.column_indices.resize(qty_light_edges);
	d_graph_light.column_indices.shrink_to_fit();
	d_graph_heavy.column_indices.resize(qty_heavy_edges);
	d_graph_heavy.column_indices.shrink_to_fit();

	d_graph_light.row_indices.resize(qty_light_edges);
	d_graph_light.row_indices.shrink_to_fit();
	d_graph_heavy.row_indices.resize(qty_heavy_edges);
	d_graph_heavy.row_indices.shrink_to_fit();

	// Partition
	thrust::stable_partition_copy(thrust::device,
			thrust::make_zip_iterator(thrust::make_tuple(
					d_graph.values.begin(),
					d_graph.column_indices.begin(),
					d_graph.row_indices.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(
					d_graph.values.end(),
					d_graph.column_indices.end(),
					d_graph.row_indices.end())),
			thrust::make_zip_iterator(thrust::make_tuple(
					d_graph_light.values.begin(),
					d_graph_light.column_indices.begin(),
					d_graph_light.row_indices.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(
					d_graph_heavy.values.begin(),
					d_graph_heavy.column_indices.begin(),
					d_graph_heavy.row_indices.begin())),
			is_light_tuple(k_delta));

	// Convert COO to CSR
	cusp::convert(d_graph_light, *d_light);
	cusp::convert(d_graph_heavy, *d_heavy);

	d_light->row_offsets.shrink_to_fit();
	d_light->column_indices.shrink_to_fit();
	d_light->values.shrink_to_fit();

	d_heavy->row_offsets.shrink_to_fit();
	d_heavy->column_indices.shrink_to_fit();
	d_heavy->values.shrink_to_fit();
}

void delta_stepping_gpu_mpgu(CudaContext& context, DCsrMatrix *d_graph, int ini_vertex, char* argv[]) {
	// Vertices
	const int k_num_vertices = d_graph->num_rows;

#if !STATS
	// Delta
	int delta = std::atoi(argv[3]);
#else
	// Read deltas
	DVector d_deltas; cusp::io::read_matrix_market_file(d_deltas, argv[3]);

	// Read ground truth
	DVector d_ground_truth;
	cusp::io::read_matrix_market_file(d_ground_truth, argv[4]);
	int truth_counter = 0, truth_checks = 0;

	// Samples
	const int k_num_samples = std::atoi(argv[5]);
	cusp::array2d<double, cusp::host_memory>  samples_sssp(d_deltas.size(), k_num_samples);

	for (int delta_idx = 0; delta_idx < d_deltas.size(); delta_idx++) {
		int delta = d_deltas[delta_idx];
		// Separate graphs according to delta
		DCsrMatrix d_graph_light, d_graph_heavy;
		separate_graphs_device(&d_graph_light, &d_graph_heavy, d_graph, delta);

		// Copy data to device
		/*
		MGPU_MEM(int) d_light_row_offsets = context.Malloc(d_graph_light.row_offsets.data(), k_num_vertices + 1);
		MGPU_MEM(int) d_light_column_indices;
		MGPU_MEM(int) d_light_values;
		if (d_graph_light.num_entries != 0)
		{
			d_light_column_indices = context.Malloc(d_graph_light.column_indices.data(), d_graph_light.num_entries);
			d_light_values = context.Malloc(d_graph_light.values.data(), d_graph_light.num_entries);
		}

		MGPU_MEM(int) d_heavy_row_offsets = context.Malloc(d_graph_heavy.row_offsets.data(), k_num_vertices + 1);
		MGPU_MEM(int) d_heavy_column_indices;
		MGPU_MEM(int) d_heavy_values;
		if (d_graph_heavy.num_entries != 0)
		{
			d_heavy_column_indices = context.Malloc(d_graph_heavy.column_indices.data(), d_graph_heavy.num_entries);
			d_heavy_values = context.Malloc(d_graph_heavy.values.data(), d_graph_heavy.num_entries);
		}*/

		for (int sample = 0; sample < k_num_samples; sample++) {
#endif
			// Timer
			struct timeval time;
			gettimeofday(&time, NULL); double t1 = time.tv_sec + (time.tv_usec / 1000000.0);

			// Set distances to infinity
			MGPU_MEM(int) d_distances = context.Fill(k_num_vertices, INT_MAX);

			// A vector which each position is the vertex id that contain the bucket it belongs to
			MGPU_MEM(int) d_vertex_id_bucket = context.Fill(k_num_vertices, INT_MAX);
			MGPU_MEM(int) d_vertex_id_bucket_dst = context.Fill(k_num_vertices, INT_MAX);

			// Counters/Offsets
			MGPU_MEM(int) d_bucket_offset = context.Fill(k_num_vertices, 0);
			MGPU_MEM(int) d_deleted_offset = context.Fill(k_num_vertices, 0);

			// Global count
			MGPU_MEM(int) d_light_count = context.Malloc<int>(k_num_vertices);
			MGPU_MEM(int) d_heavy_count = context.Malloc<int>(k_num_vertices);
			thrust::adjacent_difference(thrust::device, d_graph_light.row_offsets.begin() + 1, d_graph_light.row_offsets.end(), d_light_count->get());
			thrust::adjacent_difference(thrust::device, d_graph_heavy.row_offsets.begin() + 1, d_graph_heavy.row_offsets.end(), d_heavy_count->get());

			// Initial Variables
			int cur_bucket = 0;
			int buffer = 0;
			d_distances->FromHost(ini_vertex, sizeof(int), &buffer);
			d_vertex_id_bucket->FromHost(ini_vertex, sizeof(int), &buffer);

			// Origin vertices
			MGPU_MEM(int) d_origin_vx = context.Malloc<int>(k_num_vertices);

			while (true) {
#if DEBUG_MSG_BKT
				std::cout << "Current Bucket: " << cur_bucket << std::endl;
#endif
				// Light relaxation
				while (true) {
					// Get counts for current bucket vertices and remove them from current bucket
					thrust::for_each(thrust::device,
							thrust::make_zip_iterator(thrust::make_tuple(
									d_vertex_id_bucket->get(),
									d_vertex_id_bucket_dst->get(),
									d_light_count->get(),
									d_heavy_count->get(),
									d_bucket_offset->get(),
									d_deleted_offset->get()
							)),
							thrust::make_zip_iterator(thrust::make_tuple(
									d_vertex_id_bucket->get()     + k_num_vertices,
									d_vertex_id_bucket_dst->get() + k_num_vertices,
									d_light_count->get()          + k_num_vertices,
									d_heavy_count->get()          + k_num_vertices,
									d_bucket_offset->get()        + k_num_vertices,
									d_deleted_offset->get()       + k_num_vertices
							)),
							get_valid_bucket(cur_bucket));

					int total;
					ScanExc(d_bucket_offset->get(), k_num_vertices, &total, context);
					if (total == 0) break;

					// Get origin vertices for all outgoing edges
					LoadBalanceSearch(total, d_bucket_offset->get(), k_num_vertices, d_origin_vx->get(), context);

					thrust::for_each(thrust::device,
							thrust::make_zip_iterator(thrust::make_tuple(
									d_origin_vx->get(),
									thrust::make_counting_iterator(0)
							)),
							thrust::make_zip_iterator(thrust::make_tuple(
									d_origin_vx->get() + total,
									thrust::make_counting_iterator(total)
							)),
							update_dist(d_distances->get(),
									d_graph_light.row_offsets.data().get(),
									d_graph_light.column_indices.data().get(),
									d_graph_light.values.data().get(),
									d_bucket_offset->get(),
									d_vertex_id_bucket_dst->get(),
									delta));

					d_vertex_id_bucket.swap(d_vertex_id_bucket_dst);
				}
				// Heavy relaxation
				int total;
				ScanExc(d_deleted_offset->get(), k_num_vertices, &total, context);

				// Get origin vertices for all outgoing edges
				LoadBalanceSearch(total, d_deleted_offset->get(), k_num_vertices, d_origin_vx->get(), context);

				if (total != 0) {
					thrust::for_each(thrust::device,
							thrust::make_zip_iterator(thrust::make_tuple(
									d_origin_vx->get(),
									thrust::make_counting_iterator(0)
							)),
							thrust::make_zip_iterator(thrust::make_tuple(
									d_origin_vx->get() + total,
									thrust::make_counting_iterator(total)
							)),
							update_dist(d_distances->get(),
									d_graph_heavy.row_offsets.data().get(),
									d_graph_heavy.column_indices.data().get(),
									d_graph_heavy.values.data().get(),
									d_deleted_offset->get(),
									d_vertex_id_bucket_dst->get(),
									delta));
				}
				d_vertex_id_bucket.swap(d_vertex_id_bucket_dst);
				thrust::fill(thrust::device, d_deleted_offset->get(), d_deleted_offset->get() + k_num_vertices, 0);

				// Next bucket if there is any
				int min_bucket = thrust::reduce(thrust::device, d_vertex_id_bucket->get(),
						d_vertex_id_bucket->get() + k_num_vertices, INT_MAX, thrust::minimum<int>());
				if (min_bucket == INT_MAX) break;
				cur_bucket = min_bucket;
			}

			// Calculate elapsed time
			gettimeofday(&time, NULL); double t2 = time.tv_sec + (time.tv_usec / 1000000.0);
#if !STATS
			printf("Computation Time: %.6lf seconds\n", t2 - t1);

		    // Write distances
			thrust::device_ptr<int> p_dist(d_distances->get());
			cusp::array1d<int, cusp::device_memory>::view v_distances(p_dist, p_dist + k_num_vertices);
		    cusp::io::write_matrix_market_file(v_distances, "distance.mtx");
#else
		    // Save sample time
		    samples_sssp(delta_idx, sample) = t2 - t1;
			printf("Delta: %d, Sample #: %02d of %02d, Computation Time: %.6lf seconds\n", delta, sample + 1, k_num_samples, t2 - t1);

			// Ground truth check
			thrust::device_ptr<int> p_dist(d_distances->get());
			cusp::array1d<int, cusp::device_memory>::view v_distances(p_dist, p_dist + k_num_vertices);
			if (thrust::equal(d_ground_truth.begin(), d_ground_truth.end(), v_distances.begin())) truth_counter++;
			printf("Delta: %d, Sample #: %02d of %02d, Truth Counter: %02d of %02d\n", delta, sample + 1, k_num_samples, truth_counter, ++truth_checks);
		}
		// Write statistics every delta
		cusp::io::write_matrix_market_file(samples_sssp, "time_samples_sssp.mtx");
	}
#endif
}
