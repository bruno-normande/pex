
#include <iostream>

#include "ParticleSystem.h"
#include "ParticleSystem.cuh"
#include "helper_cuda.h"

__global__
void integrate_system(float4 *pos, float4 *vel,
						float dt, unsigned int n_particles,
						float damping)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx>=n_particles) return; 
	
	pos[idx].x = idx;
	pos[idx].y = idx*idx;

}

void ParticleSystem::integrate(){
	unsigned int n_threads, n_blocks;
	computeGridSize(n_particles,256, &n_blocks, &n_threads);
	integrate_system<<< n_blocks, n_threads >>>(dPos, dVel, dt, n_particles, 1); //TODO: Set damping
}

void ParticleSystem::copyParticlesToDevice(){
        checkCudaErrors(cudaMemcpy(dPos, hPos, sizeof(float4)*n_particles, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dVel, hVel, sizeof(float4)*n_particles, cudaMemcpyHostToDevice));

}

void ParticleSystem::copyParticlesToHost(){
        checkCudaErrors(cudaMemcpy(hPos, dPos, sizeof(float4)*n_particles, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hVel, dVel, sizeof(float4)*n_particles, cudaMemcpyDeviceToHost));

}
