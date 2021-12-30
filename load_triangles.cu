#include <stdio.h>

typedef struct SmallTriangles {
  long num_triangles;
  int small_edge_max_length;
  int* edge_lengths;
  int* triangles;
} SmallTriangles;

SmallTriangles* host_small_triangles;
SmallTriangles* device_small_triangles;


void load_small_triangles(char* filename) {
  host_small_triangles = (SmallTriangles*) malloc(sizeof(SmallTriangles));
  host_small_triangles->small_edge_max_length = host_hyperbolic_group->hyperbolicity_constant;
  FILE* fp = fopen(filename, "r");
  fscanf(fp, "%ld", &host_small_triangles->num_triangles);
  int num_triangles = host_small_triangles->num_triangles;
  int max_length = host_small_triangles->small_edge_max_length;

  host_small_triangles->edge_lengths = (int*) malloc(sizeof(int) * 3 * num_triangles);
  host_small_triangles->triangles = (int*) malloc(sizeof(int) * max_length * 4 * num_triangles);

  int triangle_index;
  for(triangle_index=0; triangle_index<num_triangles; triangle_index++) {
    int i, left_edge_length, right_edge_length, sum_edge_length, edge_index, length_index;
    // Scanning left edge
    fscanf(fp, "%d", &left_edge_length);
    edge_index = triangle_index * (4 * max_length);
    length_index = triangle_index * 3;

    for(i=0; i<left_edge_length; i++) {
      fscanf(fp, "%d", &host_small_triangles->triangles[edge_index + i]);
    }
    host_small_triangles->edge_lengths[length_index] = left_edge_length;

    // Scanning right edge
    fscanf(fp, "%d", &right_edge_length);
    edge_index = triangle_index * (4 * max_length) + max_length;
    length_index = triangle_index * 3 + 1;

    for (i=0; i<right_edge_length; i++) {
      fscanf(fp, "%d", &host_small_triangles->triangles[edge_index+i]);
    }
    host_small_triangles->edge_lengths[length_index] = right_edge_length;

    // Scanning sum edge
    fscanf(fp, "%d", &sum_edge_length);
    edge_index = triangle_index * (4 * max_length) + (2 * max_length);
    length_index = triangle_index * 3 + 2;

    for (i=0; i<sum_edge_length; i++) {
      fscanf(fp, "%d", &host_small_triangles->triangles[edge_index + i]);
    }
    host_small_triangles->edge_lengths[length_index] = sum_edge_length;
  }

  // Copying data structure to GPU now
  cudaMalloc(&device_small_triangles, sizeof(SmallTriangles));
  cudaMemcpy(&device_small_triangles->num_triangles, &host_small_triangles->num_triangles, sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(&device_small_triangles->small_edge_max_length, &host_small_triangles->small_edge_max_length, sizeof(int), cudaMemcpyHostToDevice);

  int *device_edge_lengths, *device_edges;
  cudaMalloc(&device_edge_lengths, sizeof(int) * 3 * num_triangles);
  cudaMalloc(&device_edges, sizeof(int) * max_length * 4 * num_triangles);
  cudaMemcpy(device_edge_lengths, host_small_triangles->edge_lengths, sizeof(int) * 3 * num_triangles, cudaMemcpyHostToDevice);
  cudaMemcpy(device_edges, host_small_triangles->triangles, sizeof(int) * max_length * 4 * num_triangles, cudaMemcpyHostToDevice);
  cudaMemcpy(&device_small_triangles->edge_lengths, &device_edge_lengths, sizeof(int*), cudaMemcpyHostToDevice);
  cudaMemcpy(&device_small_triangles->triangles, &device_edges, sizeof(int*), cudaMemcpyHostToDevice);
}
