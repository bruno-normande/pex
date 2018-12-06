/*
 * DirectChecking_cuda.cu
 *
 *  Created on: 02/02/2016
 *      Author: bruno
 */

#include <cuda.h>

#include "DirectChecking.h"
#include "ParticleSystem.h"
#include "World.cuh"
#include "aux.h"

extern __constant__
SysParams system_params;

__global__
void calculate_contact_force(thrust::host_vector<float4>& dPos, thrust::host_vector<float4>& dVel, 
                                thrust::host_vector<float4>& dFor){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int n_particles = dPos.size();
        if(idx>=n_particles) return;

        float3 force = make_float3(0,0,0);
        float r = system_params.particle_radius;
        float3 position = make_float3(dPos[idx]);
        float3 velocity = make_float3(dVel[idx]);
        for(int i = 0; i < n_particles; i++){
                if(i != idx)
                    force += World::contactForce(position, make_float3(dPos[i]), velocity, make_float3(dVel[i]), r, r);
        }
        dFor[idx] = make_float4(force, 0);
}

void DirectChecking::calculateContactForce(thrust::host_vector<float4>& dPos, thrust::host_vector<float4>& dVel, 
                                                thrust::host_vector<float4>& dFor){
	unsigned int n_threads, n_blocks;
        unsigned int n_particles = dPos.size();
	computeGridSize(n_particles,256, &n_blocks, &n_threads);
	calculate_contact_force<<<n_blocks, n_threads>>>(dPos, dVel, dFor);
}


