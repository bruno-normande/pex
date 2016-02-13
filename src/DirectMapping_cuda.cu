/*
 * DirectMapping_cuda.cu
 *
 *  Created on: 09/02/2016
 *      Author: bruno
 */

#include "DirectMapping.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "aux.h"
#include "World.cuh"
#include <cuda.h>

extern __constant__
SysParams system_params;

__global__
void create_neighboor_grid(float4 *pos, int *grid_list, int *grid_count,
		unsigned int n_particles, float3 p_min, float d, int3 gridDim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= n_particles)
		return;

	int3 gridPos = get_grid_pos(make_float3(pos[idx]), p_min, d);

	int grid_idx = pos_to_index(gridPos, gridDim);

	int list_idx = atomicAdd(&grid_count[grid_idx], 1);
	grid_list[grid_idx*CELL_MAX_P + list_idx] = idx;

}

void DirectMapping::createNeighboorList(float4 *dPos, float4 *dVel){
	checkCudaErrors(cudaMemset(dGridCounter, 0, sizeof(int) * gridDim.x * gridDim.y * gridDim.z));

	unsigned int numBlocks, numThreads;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	create_neighboor_grid<<<numBlocks, numThreads>>>(dPos, dGrid, dGridCounter,
			n_particles, p_min, d, gridDim);

	getLastCudaError("Kernel execution failed: create_neighboor_grid");
}

__global__
void dm_calculate_contact_force(int *grid_list, int *grid_count, float4 *pos,
		float4 *vel, float4 *force, unsigned int n_particles, float3 pMin,
		float d, int3 gridDim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= n_particles)
		return;


	float3 resulting_force = make_float3(0);
    float3 my_pos = make_float3(pos[idx]);
    float3 my_vel = make_float3(vel[idx]);

    int3 gridPos = get_grid_pos(my_pos, pMin, d);
	int grid_idx = pos_to_index(gridPos, gridDim);
	int r = system_params.particle_radius;

	for(int z = -1; z <= 1; z++){
		for(int y = -1; y <= 1; y++){
			for(int x = -1; x <= 1; x++){
				// check boundaries
				int3 other_cell_pos = gridPos + make_int3(x,y,z);
				if(other_cell_pos.x < 0 || other_cell_pos.y < 0 ||
						other_cell_pos.z < 0 ||other_cell_pos.x >= gridDim.x ||
						other_cell_pos.y >= gridDim.y ||
						other_cell_pos.x >= gridDim.z)
				{
					continue;
				}

				int other_cell = pos_to_index(other_cell_pos, gridDim);
				for(int i = 0; i < grid_count[other_cell]; i++){
					int p_index = grid_list[other_cell*CELL_MAX_P + i - 1];
					if(p_index != idx)
						resulting_force += World::contactForce(my_pos,
													make_float3(pos[p_index]),
													my_vel,
													make_float3(vel[p_index]),
													r, r);
				}
			}
		}
	}
	force[idx] = make_float4(resulting_force);

}

void DirectMapping::calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor){
	unsigned int numBlocks, numThreads;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	dm_calculate_contact_force<<<numBlocks, numThreads>>>(dGrid, dGridCounter,
			dPos, dVel, dFor, n_particles, p_min, d, gridDim);

	getLastCudaError("Kernel execution failed: dm_calculate_contact_force");
}

