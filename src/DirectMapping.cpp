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

DirectMapping::DirectMapping() :
	dGrid(NULL), dList(NULL), d(0), n_particles(0)
{ }

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

void DirectMapping::setParams(SysParams params){
	d = params.particle_radius;
	n_particles = params.n_particles;
	gridDim.x = ceil( (params.p_max.x - params.p_min.x) / d);
	gridDim.y = ceil( (params.p_max.y - params.p_min.y) / d);
	gridDim.z = ceil( (params.p_max.z - params.p_min.z) / d);
}

std::string DirectMapping::getName(){
	return std::string("Direct Mapping");
}

