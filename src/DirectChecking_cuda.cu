/*
 * DirectChecking_cuda.cu
 *
 *  Created on: 02/02/2016
 *      Author: bruno
 */

#include <cuda.h>

#include "DirectChecking.h"
#include "World.cuh"
#include "aux.h"

__constant__
SysParams system_params;

void DirectChecking::calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor){
	unsigned int n_threads, n_blocks;
	computeGridSize(n_particles,256, &n_blocks, &n_threads);
	calculate_contact_force<<<n_blocks, n_threads>>>(dPos, dFor, n_particles);
}

__global__
void calculate_contact_force(float4 *dPos, float4 *dFor, unsigned int n_particles){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx>=n_particles) return;

	float3 force = make_float3(0,0,0);
	float r = system_params.particle_radius;
	for(int i = 0; i < n_particles; i++){
		force += World::contactForce(dPos[idx], dPos[i], dVel[idx], dVel[i], r, r);
	}
	dFor[idx] = make_float4(force, 0);
}


