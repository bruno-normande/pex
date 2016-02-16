/*
 * CellMapping.cpp
 *
 *  Created on: 01/02/2016
 *      Author: bruno
 */

#include "CellMapping.h"

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "ParticleSystem.h"

CellMapping::CellMapping() :
	dGrid(NULL), dGridCounter(NULL), d(0), n_particles(0)
{
	p_min = make_float3(0);
}

CellMapping::~CellMapping() {
	if(dGrid)
		cudaFree(dGrid);
	if(dGridCounter)
		cudaFree(dGridCounter);
}

void CellMapping::memInitialize(){
	checkCudaErrors(cudaMalloc((void**) &dGrid, sizeof(int) * gridDim.x * gridDim.y * gridDim.z * CELL_MAX_P));
	checkCudaErrors(cudaMalloc((void**) &dGridCounter, sizeof(int) * gridDim.x * gridDim.y * gridDim.z));
}

void CellMapping::setParams(SysParams params){
	d = params.particle_radius;
	n_particles = params.n_particles;
	gridDim.x = ceil( (params.p_max.x - params.p_min.x) / d);
	gridDim.y = ceil( (params.p_max.y - params.p_min.y) / d);
	gridDim.z = ceil( (params.p_max.z - params.p_min.z) / d);
	p_min = params.p_min;
}

std::string CellMapping::getName(){
	return std::string("Cell Mapping");
}

