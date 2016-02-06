/*
 * SortingContactDetection.cpp
 *
 *  Created on: 04/02/2016
 *      Author: bruno
 */

#include "SortingContactDetection.h"
#include "helper_math.h"

#include <cuda.h>
#include "helper_cuda.h"

SortingContactDetection::SortingContactDetection() :
	dSortedPos(NULL), dSortedVel(NULL),
	dGridParticleHash(NULL), dGridParticleIndex(NULL),
	dCellStart(NULL), dCellEnd(NULL),
	n_particles(0), d(0)
{
	gridSortBits = 18;    // increase this for larger grids ??
	p_max = make_float3(0);
	p_min = make_float3(0);
	gridSize = make_int3(0);

}

SortingContactDetection::~SortingContactDetection() {
	if(dSortedPos)
		cudaFree(dSortedPos);
	if(dSortedVel)
		cudaFree(dSortedVel);
	if(dGridParticleHash)
		cudaFree(dGridParticleHash);
	if(dGridParticleIndex)
		cudaFree(dGridParticleIndex);
	if(dCellStart)
		cudaFree(dCellStart);
	if(dCellEnd)
		cudaFree(dCellEnd);

}

void SortingContactDetection::memInitialize(){
	checkCudaErrors(cudaMalloc((void**) &dSortedPos, sizeof(float4) * n_particles));
	checkCudaErrors(cudaMalloc((void**) &dSortedVel, sizeof(float4) * n_particles));

	checkCudaErrors(cudaMalloc((void**) &dGridParticleHash, sizeof(unsigned int) * n_particles));
	checkCudaErrors(cudaMalloc((void**) &dGridParticleIndex, sizeof(unsigned int) * n_particles));

	int grid_size = gridSize.x*gridSize.y*gridSize.z;
	checkCudaErrors(cudaMalloc((void**) &dCellStart, sizeof(unsigned int) * grid_size));
	checkCudaErrors(cudaMalloc((void**) &dCellEnd, sizeof(unsigned int) * grid_size));

}

void SortingContactDetection::createNeighboorList(float4 *dPos, float4 *dVel){
	// calc hash
	calcHash(dPos);

	// sortparticles
	sortParticles();

	// reorder data and find cell start
	reorderAndSetStart(dPos, dVel);
}

void SortingContactDetection::setParams(SysParams params){
	n_particles = params.n_particles;
	p_max = params.p_max;
	p_min = params.p_min;
	d = params.particle_radius;

	gridSize.x = ceil((p_max.x - p_min.x)/d);
	gridSize.y = ceil((p_max.y - p_min.y)/d);
	gridSize.z = ceil((p_max.z - p_min.z)/d);

}
