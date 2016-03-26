/*
 * SortingAndSearch.cpp
 *
 *  Created on: 19/03/2016
 *      Author: bruno
 */

#include "SortingAndSearch.h"
#include "helper_math.h"

#include <cuda.h>
#include "helper_cuda.h"

SortingAndSearch::SortingAndSearch() :
	dSortedPos(NULL), dSortedVel(NULL),
	dSortedGrid(NULL), n_particles(0), d(0)
{
	p_max = make_float3(0);
	p_min = make_float3(0);
	gridSize = make_int3(0);

}

SortingAndSearch::~SortingAndSearch() {
	if(dSortedPos)
		cudaFree(dSortedPos);
	if(dSortedVel)
		cudaFree(dSortedVel);
	if(dSortedGrid)
		cudaFree(dSortedGrid);

}

void SortingAndSearch::memInitialize(){
	checkCudaErrors(cudaMalloc((void**) &dSortedPos, sizeof(float4) * n_particles));
	checkCudaErrors(cudaMalloc((void**) &dSortedVel, sizeof(float4) * n_particles));

	checkCudaErrors(cudaMalloc((void**) &dSortedGrid, sizeof(uint4) * n_particles));
}

void SortingAndSearch::createNeighboorList(float4 *dPos, float4 *dVel){
	// pre
	prepareGrid(dPos);

	// sortparticles
	sortParticles();

	// reorder data and find cell start
	reorderPosAndVel(dPos, dVel);
}

void SortingAndSearch::setParams(SysParams params){
	n_particles = params.n_particles;
	p_max = params.p_max;
	p_min = params.p_min;
	d = params.particle_radius;

	gridSize.x = ceil((p_max.x - p_min.x)/d);
	gridSize.y = ceil((p_max.y - p_min.y)/d);
	gridSize.z = ceil((p_max.z - p_min.z)/d);

}
