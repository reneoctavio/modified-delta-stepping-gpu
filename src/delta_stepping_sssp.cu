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
 * Calculate the shortest path given the graph, a initial vertex and the output distance
 * @param d_graph_light a graph with light edges (in)
 * @param d_graph_heavy a graph with heavy edges (in)
 * @param d_distance a vector with the distance from the initial vertex to all other vertices (out)
 * @param k_delta a constant delta
 * @param ini_vertex the initial vertex
 */
void delta_stepping_gpu_sssp(DCsrMatrix *d_graph_light, DCsrMatrix *d_graph_heavy, DVector *d_distance, const int k_delta, int ini_vertex) {
	// Vertices
	const int k_num_vertices = d_graph_light->num_rows;

	// Set distances to infinity
	thrust::fill(d_distance->begin(), d_distance->end(), INT_MAX);

	// A vector which each position is the vertex id that contain the bucket it belongs to
	DVector vertex_id_bucket(k_num_vertices, INT_MAX);

	// A vector with deleted vertices in the light relaxation phase
	DVector deleted_vertices(k_num_vertices, 0);

	// Initial Variables
	int cur_bucket = 0;
	vertex_id_bucket[ini_vertex] = 0;
	(*d_distance)[ini_vertex] = 0;

	// Count of outgoing edges -- Light
	DVector d_out_edges_count_light(k_num_vertices + 1);
	thrust::adjacent_difference(thrust::device, d_graph_light->row_offsets.begin(),
			d_graph_light->row_offsets.end(), d_out_edges_count_light.begin());
	d_out_edges_count_light.erase(d_out_edges_count_light.begin());
	d_out_edges_count_light.shrink_to_fit();

	// Count of outgoing edges -- Heavy
	DVector d_out_edges_count_heavy(k_num_vertices + 1);
	thrust::adjacent_difference(thrust::device, d_graph_heavy->row_offsets.begin(),
			d_graph_heavy->row_offsets.end(), d_out_edges_count_heavy.begin());
	d_out_edges_count_heavy.erase(d_out_edges_count_heavy.begin());
	d_out_edges_count_heavy.shrink_to_fit();

	// Bucket stencil (holds where elements are)
	DVector bucket_stencil(k_num_vertices);

	// STATS
#if STATS_WORK
	DVector bucket_vertices;
	DVector edges_touched;
	DVector queued_vertices;
#endif

	while (1) {
		// Create stencil
		thrust::fill(bucket_stencil.begin(), bucket_stencil.end(), 0);

		// Every vertex that belongs to this bucket will have 1 in the stencil
		thrust::transform(vertex_id_bucket.begin(), vertex_id_bucket.end(), bucket_stencil.begin(), is_inside_bucket(cur_bucket));
		int bucket_sz = thrust::reduce(bucket_stencil.begin(), bucket_stencil.end());

		// Copy vertices to current bucket
		DVector d_bucket(bucket_sz);
		thrust::copy_if(thrust::make_counting_iterator(0),
				thrust::make_counting_iterator(k_num_vertices),
				bucket_stencil.begin(), d_bucket.begin(), thrust::identity<int>());

		// Vector with removed vertices
		DVector d_removed;

		// STATS
#if STATS_WORK
		bucket_vertices.push_back(d_bucket.size());
		edges_touched.push_back(0);
		int queued_num = thrust::count_if(vertex_id_bucket.begin(), vertex_id_bucket.end(), is_valid_bucket()) - bucket_sz;
		queued_vertices.push_back(queued_num);
#endif

#if DEBUG_MSG_BKT
		std::cout << "Current Bucket: " << cur_bucket << std::endl;
#endif
		// Light relaxation
		while (bucket_sz != 0) {
			// Remove origin vertices from buckets
			thrust::fill(
					thrust::make_permutation_iterator(
							vertex_id_bucket.begin(),
							d_bucket.begin()),
					thrust::make_permutation_iterator(
							vertex_id_bucket.begin(),
							d_bucket.end()),
					INT_MAX);

			// Update frontier and distances
			DVector d_frontier_vertices;
			DVector d_tent_distance;
			expand_edges(d_graph_light, d_distance,
					&d_out_edges_count_light, &d_bucket,
					bucket_sz, &d_frontier_vertices, &d_tent_distance);

			// STATS
#if STATS_WORK
			edges_touched[edges_touched.size() - 1] += d_frontier_vertices.size();
#endif

			// Remove tentative > current distance
			remove_invalid_distances(d_distance, &d_frontier_vertices, &d_tent_distance);

			// Update distance and buckets
			relax(d_distance, &vertex_id_bucket, &d_frontier_vertices, &d_tent_distance, k_delta);

			// Copy vertices to removed
			d_removed.resize(d_removed.size() + bucket_sz);

			// Copy to queue vertices not in queue already
			DVectorIterator last_queued = thrust::copy_if(thrust::device, d_bucket.begin(), d_bucket.end(),
					(d_removed.end() - d_bucket.size()), copy_non_deleted(deleted_vertices.begin()));

			d_removed.erase(last_queued, d_removed.end());

			// Update deleted map
			thrust::fill(
					thrust::make_permutation_iterator(
							deleted_vertices.begin(),
							d_bucket.begin()),
					thrust::make_permutation_iterator(
							deleted_vertices.begin(),
							d_bucket.end()),
					1);

			// Update stencil
			thrust::fill(bucket_stencil.begin(), bucket_stencil.end(), 0);
			thrust::transform(vertex_id_bucket.begin(), vertex_id_bucket.end(), bucket_stencil.begin(), is_inside_bucket(cur_bucket));
			bucket_sz = thrust::reduce(bucket_stencil.begin(), bucket_stencil.end());

			// Copy new vertices to this bucket
			d_bucket.resize(bucket_sz);
			thrust::copy_if(thrust::make_counting_iterator(0),
					thrust::make_counting_iterator(k_num_vertices),
					bucket_stencil.begin(), d_bucket.begin(), thrust::identity<int>());

			// STATS
#if STATS_WORK
			bucket_vertices[bucket_vertices.size() - 1] += d_bucket.size();
#endif
		}
		// Heavy relaxation
		// Update frontier and distances
		DVector d_frontier_vertices;
		DVector d_tent_distance;
		expand_edges(d_graph_heavy, d_distance,
				&d_out_edges_count_heavy, &d_removed,
				d_removed.size(), &d_frontier_vertices, &d_tent_distance);

		// STATS
#if STATS_WORK
		edges_touched[edges_touched.size() - 1] += d_frontier_vertices.size();
#endif

		// Remove tentative > current distance
		remove_invalid_distances(d_distance, &d_frontier_vertices, &d_tent_distance);

		// Update distance and buckets
		relax(d_distance, &vertex_id_bucket, &d_frontier_vertices, &d_tent_distance, k_delta);

		// Next bucket if there is any
		DVectorIterator min_bucket = thrust::min_element(vertex_id_bucket.begin(), vertex_id_bucket.end());
		if (*min_bucket == INT_MAX) break;
		cur_bucket = *min_bucket;
	}
#if STATS_WORK
	// Write statistics
	cusp::io::write_matrix_market_file(bucket_vertices, "bucket_vertices.mtx");
	cusp::io::write_matrix_market_file(edges_touched, "edges_touched.mtx");
	cusp::io::write_matrix_market_file(queued_vertices, "queued_vertices.mtx");
#endif
}

/**
 * Given an origin vertex, expand it to get their edges and frontier vertices
 * @param d_graph a graph to be traversed (in)
 * @param d_distance a vector holding the current distance from initial vertex to others (in)
 * @param d_origin a vector with origin vertices (in)
 * @param num_origin_vertices number of origin vertices (in)
 * @param d_frontier_vertices a vector with frontier vertices (out)
 * @param d_tentative_distance a vector with tentative distance of these frontier edges (out)
 */
void expand_edges(DCsrMatrix *d_graph, DVector *d_distance, DVector *d_count, DVector *d_origin,
		int num_origin_vertices, DVector *d_frontier_vertices, DVector *d_tentative_distance) {

	// Calculate the offsets, where the frontier vertices and tentative distances will be placed
	DVector edges_offset(num_origin_vertices + 1, 0);
	thrust::copy(thrust::device,
			thrust::make_permutation_iterator(
					d_count->begin(),
					d_origin->begin()),
			thrust::make_permutation_iterator(
					d_count->begin(),
					d_origin->begin() + num_origin_vertices),
			edges_offset.begin());
	thrust::exclusive_scan(thrust::device, edges_offset.begin(), edges_offset.end(), edges_offset.begin());

	int num_out_edges = edges_offset.back();
	d_frontier_vertices->resize(num_out_edges);
	d_tentative_distance->resize(num_out_edges);

	// Copy frontier vertices and tentative distances from graph and others input
	thrust::for_each(thrust::device,
			make_zip_iterator(thrust::make_tuple(
					thrust::make_permutation_iterator(
						d_graph->row_offsets.begin(),
						d_origin->begin()),
					thrust::make_permutation_iterator(
						d_distance->begin(),
						d_origin->begin()),
					edges_offset.begin(),
					edges_offset.begin() + 1)),
			make_zip_iterator(thrust::make_tuple(
					thrust::make_permutation_iterator(
						d_graph->row_offsets.begin(),
						d_origin->begin() + num_origin_vertices),
					thrust::make_permutation_iterator(
						d_distance->begin(),
						d_origin->begin() + num_origin_vertices),
					edges_offset.begin() + num_origin_vertices,
					edges_offset.begin() + 1 + num_origin_vertices)),
			update_distance_and_vertices(d_tentative_distance->data().get(),
										 d_frontier_vertices->data().get(),
										 d_graph->column_indices.data().get(),
										 d_graph->values.data().get()));
}

/**
 * Remove tentative distances if they are greater or equal of current distance of a frontier vertex
 * Also remove associated frontier vertex
 * @param d_distance a vector with current distances (in)
 * @param d_frontier_vertices a vector with current frontier vertices (in/out)
 * @param d_tent_distance a vector with tentative distances (in/out)
 */
void remove_invalid_distances(DVector *d_distance, DVector *d_frontier_vertices, DVector *d_tent_distance) {
	Zip2DVecIterator zip_removed_end;
	Tuple2DVec tent_dist_iter;
	zip_removed_end = thrust::remove_if(thrust::device,
			thrust::make_zip_iterator(
					thrust::make_tuple(
							d_frontier_vertices->begin(),
							d_tent_distance->begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							d_frontier_vertices->end(),
							d_tent_distance->end())),
			is_invalid_tent_distance(d_distance->data().get()));

	tent_dist_iter = zip_removed_end.get_iterator_tuple();
	d_frontier_vertices->erase(thrust::get<0>(tent_dist_iter), d_frontier_vertices->end());
	d_tent_distance->erase(thrust::get<1>(tent_dist_iter), d_tent_distance->end());
}

/**
 * Relaxation phase
 * Update distance with the minimum tentative distance and also update frontier vertices buckets
 * @param d_distance a vector with current distances (in/out)
 * @param vertex_id_bucket a vector with buckets of all vertices (in/out)
 * @param d_frontier_vertices a vector with current frontier vertices (in)
 * @param d_tent_distance a vector with tentative distances (in)
 * @param k_delta a constant delta
 */
void relax(DVector *d_distance, DVector *vertex_id_bucket, DVector *d_frontier_vertices, DVector *d_tent_distance, const int k_delta) {
	thrust::for_each(thrust::device,
			thrust::make_zip_iterator(thrust::make_tuple(
					d_tent_distance->begin(),
					thrust::make_permutation_iterator(
							d_distance->begin(),
							d_frontier_vertices->begin()),
					thrust::make_permutation_iterator(
							vertex_id_bucket->begin(),
							d_frontier_vertices->begin())
					)),
			thrust::make_zip_iterator(thrust::make_tuple(
					d_tent_distance->end(),
					thrust::make_permutation_iterator(
							d_distance->begin(),
							d_frontier_vertices->end()),
					thrust::make_permutation_iterator(
							vertex_id_bucket->begin(),
							d_frontier_vertices->end())
					)),
			update_distance(k_delta));
}

/**
 * Separate light and heavy graphs based on the original graph
 * Edges are light if less and equal to delta, and heavy otherwise
 * @param d_light a light CSR graph (out)
 * @param d_heavy a heavy CSR graph (out)
 * @param d_total_graph the complete CSR graph (in)
 * @param k_delta a constant delta
 */
void separate_graphs(DCsrMatrix *d_light, DCsrMatrix *d_heavy, DCsrMatrix *d_total_graph, const int k_delta)
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

