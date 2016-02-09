/*
 * DirectMapping_cuda.cu
 *
 *  Created on: 09/02/2016
 *      Author: bruno
 */

#include "DirectMapping.h"
#include "aux.h"


__global__
void create_neighboor_grid(float4 *pos, int *grid, unsigned int n_particles,
		float3 p_min, float d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

void DirectMapping::createNeighboorList(float4 *dPos, float4 *dVel){
	checkCudaErrors(cudaMemset(dGrid, EMPTY, sizeof(int) * gridDim.x * gridDim.y * gridDim.z));

	unsigned int numBlocks, numThreads;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	create_neighboor_grid<<<numBlocks, numThreads>>>(dPos, dGrid, n_particles, p_min, d);

	getLastCudaError("Kernel execution failed: create_neighboor_grid");
}

