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

__global__
void calc_hash(unsigned int *dGridParticleHash, unsigned int *dGridParticleIndex,
		float4 *dPos, unsigned int n_particles, float4 p_min, float d){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx>=n_particles) return;

	int3 gridPos = get_grid_pos(dPos[idx], p_min, d);
	unsigned int hash = gridPos.x*gridPos.y*gridPos.z + gridPos.x*gridPos.y + gridPos.x;

	dGridParticleHash[idx] = hash;
	dGridParticleIndex[idx] = idx;

}


void SortingContactDetection::calcHash(float4 *dPos){
	unsigned int numThreads, numBlocks;
	computeGridSize(n_particles, 256, &numBlocks, &numThreads);

	calc_hash<<<numBlocks, numThreads>>>(dGridParticleHash, dGridParticleIndex,
			dPos, n_particles, p_min, d);

	getLastCudaError("Kernel execution failed");
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
	checkCudaErrors(cudaMemset(dCellStart, 0xffffffff, numCells*sizeof(uint)));
	checkCudaErrors(cudaMemset(dCellEnd, 0xffffffff, numCells*sizeof(uint)));

	unsigned int shared_mem = sizeof(unsigned int)*(numThreads+1);
	reorder_and_find_cell_start<<<numThreads, numBlocks, shared_mem>>>(dCellStart, dCellEnd,
			dSortedPos, dSortedVel, dGridParticleHash, dGridParticleIndex, dPos,
			dVel, n_particles);

	getLastCudaError("Kernel execution failed");
}






