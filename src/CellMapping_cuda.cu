/*
 * CellMapping_cuda.cu
 *
 *  Created on: 09/02/2016
 *      Author: bruno
 */

#include "CellMapping.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "aux.h"
#include "World.cuh"
#include <cuda.h>

extern __constant__
SysParams system_params;

__global__
void cm_create_neighboor_grid(float4 *pos, int *grid_list, int *grid_count,
		unsigned int n_particles, float3 p_min, float d, int3 gridDim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= n_particles)
		return;

	float r = system_params.particle_radius;
	//Upper Righe Back
	int3 gridPosURB = get_grid_pos(make_float3(pos[idx]) + make_float3(r), p_min, d);
	// Down Left Front
	int3 gridPosDLF = get_grid_pos(make_float3(pos[idx]) + make_float3(-r), p_min, d);

	for(int i = gridPosDLF.x; i <= gridPosURB.x; i++){
		for(int j = gridPosDLF.y; j <= gridPosURB.y; j++){
			for(int k = gridPosDLF.z; k <= gridPosURB.z; k++){
				int3 cell = make_int3(i,j,k);

				// check boundaries
				if(cell.x < 0 || cell.y < 0 || cell.z < 0 ||
						cell.x >= gridDim.x || cell.y >= gridDim.y ||
						cell.z >= gridDim.z)
				{
					continue;
				}

				int cell_idx = pos_to_index(cell, gridDim);
				int list_idx = atomicAdd(&grid_count[cell_idx], 1);
				grid_list[cell_idx*CELL_MAX_P + list_idx] = idx;
			}
		}
	}

}

void CellMapping::createNeighboorList(float4 *dPos, float4 *dVel){
	checkCudaErrors(cudaMemset(dGridCounter, 0, sizeof(int) * gridDim.x * gridDim.y * gridDim.z));

	unsigned int numBlocks, numThreads;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	cm_create_neighboor_grid<<<numBlocks, numThreads>>>(dPos, dGrid, dGridCounter,
			n_particles, p_min, d, gridDim);

	getLastCudaError("Kernel execution failed: create_neighboor_grid");
}

__global__
void cm_calculate_contact_force(int *grid_list, int *grid_count, float4 *pos,
		float4 *vel, float4 *force, unsigned int n_particles, float3 pMin,
		float d, int3 gridDim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= n_particles)
		return;


	float3 resulting_force = make_float3(0);
    float3 my_pos = make_float3(pos[idx]);
    float3 my_vel = make_float3(vel[idx]);

	float r = system_params.particle_radius;

	//Upper Righe Back
	int3 gridPosURB = get_grid_pos(my_pos + make_float3(r), pMin, d);
	// Down Left Front
	int3 gridPosDLF = get_grid_pos(my_pos + make_float3(-r), pMin, d);

	for(int i = gridPosDLF.x; i <= gridPosURB.x; i++){
		for(int j = gridPosDLF.y; j <= gridPosURB.y; j++){
			for(int k = gridPosDLF.z; k <= gridPosURB.z; k++){
				// check boundaries
				int3 cell = make_int3(i,j,k);
				if(cell.x < 0 || cell.y < 0 || cell.z < 0 || cell.x >= gridDim.x
						|| cell.y >= gridDim.y || cell.z >= gridDim.z)
				{
					continue;
				}

				int cell_idx = pos_to_index(cell, gridDim);
				for(int i = 0; i < grid_count[cell_idx]; i++){
					int p_index = grid_list[cell_idx*CELL_MAX_P + i];
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

void CellMapping::calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor){
	unsigned int numBlocks, numThreads;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	cm_calculate_contact_force<<<numBlocks, numThreads>>>(dGrid, dGridCounter,
			dPos, dVel, dFor, n_particles, p_min, d, gridDim);

	getLastCudaError("Kernel execution failed: dm_calculate_contact_force");
}

