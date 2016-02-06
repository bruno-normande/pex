/*
 * SortingContactDetection.cu
 *
 *  Created on: 05/02/2016
 *      Author: bruno
 */

#include "SortingContactDetection.h"
#include "aux.h"
#include "helper_cuda.h"

#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define EMPTY 0xffffffff

extern __constant__
SysParams system_params;

__device__
unsigned int makeHash(int3 pos){
	return pos.x*pos.y*pos.z + pos.x*pos.y + pos.x;
}

__global__
void calc_hash(unsigned int *dGridParticleHash, unsigned int *dGridParticleIndex,
		float4 *dPos, unsigned int n_particles, float4 p_min, float d){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx>=n_particles) return;

	int3 gridPos = get_grid_pos(dPos[idx], p_min, d);
	unsigned int hash = makeHash(gridPos);

	dGridParticleHash[idx] = hash;
	dGridParticleIndex[idx] = idx;

}


void SortingContactDetection::calcHash(float4 *dPos){
	unsigned int numThreads, numBlocks;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	calc_hash<<<numBlocks, numThreads>>>(dGridParticleHash, dGridParticleIndex,
			dPos, n_particles, p_min, d);

	getLastCudaError("Kernel execution failed: calc_hash");
}

void SortingContactDetection::sortParticles(){
	thrust::sort_by_key(thrust::device_ptr<unsigned int>(dGridParticleHash),
			thrust::device_ptr<unsigned int>(dGridParticleHash + n_particles),
			thrust::device_ptr<unsigned int>(dGridParticleIndex));
}

__global__
void reorder_and_find_cell_start(unsigned int *cellStart, unsigned int *cellEnd,
        float *sortedPos, float *sortedVel, unsigned int *gridParticleHash,
        unsigned int *gridParticleIndex, float *oldPos, float4 *oldVel,
        unsigned int n_particles)
{
	extern __shared__ unsigned int sharedHash[]; //TODO: Alocar e desalocar

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int hash;
	if(index < n_particles){
		hash = gridParticleHash[idx];

		sharedHash[threadIdx.x + 1] = hash;

		if(idx > 0 && threadIdx.x == 0){
			sharedHash[0] = gridParticleHash[idx-1];
		}
	}

	__syncthreads();

	if(idx < n_particles){
		if(idx == 0 || hash != sharedHash[threadIdx.x]){
			cellStart[hash] = idx;

			if(idx > 0){
				cellEnd[sharedHash[threadIdx.x]] = idx;
			}
		}

		if(idx == n_particles - 1){
			cellEnd[hash] = idx + 1;
		}

		// aproveita para pegar os valores da posição e velocidades ordenados
		unsigned int sortedIdx = gridParticleIndex[idx];
		sortedPos[idx] = oldPos[sortedIdx];
		sortedVel[idx] = oldVel[sortedIdx];
	}
}

void SortingContactDetection::reorderAndSetStart(float4 *dPos, float4 *dVel){
	unsigned int numThreads, numBlocks;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	unsigned int numCells = gridSize.x*gridSize.y*gridSize.z;
	checkCudaErrors(cudaMemset(dCellStart, EMPTY, numCells*sizeof(uint)));
	checkCudaErrors(cudaMemset(dCellEnd, EMPTY, numCells*sizeof(uint)));

	unsigned int shared_mem = sizeof(unsigned int)*(numThreads+1);
	reorder_and_find_cell_start<<<numThreads, numBlocks, shared_mem>>>(dCellStart, dCellEnd,
			dSortedPos, dSortedVel, dGridParticleHash, dGridParticleIndex, dPos,
			dVel, n_particles);

	getLastCudaError("Kernel execution failed: reorder_and_find_cell_start");
}

__global__
void calculate_contact_force(float4 *sortedPos, float4 sortedVel,
		unsigned int *gridParticleIndex, unsigned int *cellStart,
		unsigned int *cellEnd, float4 *force, unsigned int n_particles,
		float3 pMin, float d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= n_particles) return;

	float3 pos = make_float3(sortedPos[idx]);
	float3 vel = make_float3(sortedVel[idx]);

	int3 gridPos = get_grid_pos(pos, pMin, d);

	float3 resulting_force = make_float3(0);

	float r = system_params.particle_radius;

	for(int z = -1; z <= 1; z++){
		for(int y = -1; y <= 1; y++){
			for(int x = -1; x <= 1; x++){
				unsigned int hash =  makeHash(gridPos + make_float3(x,y,z));
				unsigned int start_idx = cellStart[hash];

				if(start_idx != EMPTY){
					unsigned int end_idx = cellEnd[hash];

					for(unsigned int i=start_idx; i< end_idx; i++){
						if(i == idx) continue; // jumps self
						float3 neigh_pos = make_float3(sortedPos[i]);
						float3 neigh_vel = make_float3(sortedVel[i]);

						resulting_force += World::contactForce(pos, neigh_pos,
								vel, neigh_vel, r, r);
					}
				}
			}
		}
	}
}

void SortingContactDetection::calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor){
	// will not use dPos and dVel since i have my own version stored
	unsigned int numThreads, numBlocks;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	calculate_contact_force<<<numThreads, numBlocks>>>(dSortedPos, dSortedVel,
			dGridParticleIndex, dCellStart, dCellEnd, dFor, n_particles, p_min, d);
	getLastCudaError("Kernel execution failed: calculate_contact_force");
}



