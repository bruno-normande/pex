/*
 * DirectMapping.cpp
 *
 *  Created on: 01/02/2016
 *      Author: bruno
 */

#include "DirectMapping.h"

#include <cuda.h>
#include "helper_cuda.h"

DirectMapping::DirectMapping(float3 pMax, float3 pMin, float d, unsigned int n_particle) :
	dGrid(NULL), dList(NULL), n_particle(n_particle)
{
	griDim.x = ceil( (pMax.x - pMin.x) / d);
	griDim.y = ceil( (pMax.y - pMin.y) / d);
	griDim.z = ceil( (pMax.z - pMin.z) / d);

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

void DirectMapping::calculateContactForce(){

}
