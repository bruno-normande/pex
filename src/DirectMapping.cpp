/*
 * DirectMapping.cpp
 *
 *  Created on: 01/02/2016
 *      Author: bruno
 */

#include "DirectMapping.h"

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "ParticleSystem.h"

DirectMapping::DirectMapping(unsigned int n_particle, SysParams params) :
	dGrid(NULL), dList(NULL), n_particles(n_particle)
{
	d = params.particle_radius;
}

DirectMapping::~DirectMapping() {
	if(dGrid)
		cudaFree(dGrid);
	if(dList)
		cudaFree(dList);
}

void DirectMapping::memInitialize(){
	checkCudaErrors(cudaMalloc((void**) &dGrid, sizeof(int) * gridDim.x * gridDim.y * gridDim.z));
	checkCudaErrors(cudaMalloc((void**) &dList, sizeof(int) * n_particles));
}

void DirectMapping::createNeighboorList(float4 *dPos){
	checkCudaErrors(cudaMemset(dGrid, 0, sizeof(int) * gridDim.x * gridDim.y * gridDim.z));
	checkCudaErrors(cudaMemset(dList, 0, sizeof(int) * n_particles));

}

void DirectMapping::calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor){

}

void DirectMapping::setMinMax(float3 pMin, float3 pMax){
	gridDim.x = ceil( (pMax.x - pMin.x) / d);
	gridDim.y = ceil( (pMax.y - pMin.y) / d);
	gridDim.z = ceil( (pMax.z - pMin.z) / d);
}

std::string DirectMapping::getName(){
	return std::string("Direct Mapping");
}

