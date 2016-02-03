
#include <iostream>

#include "ParticleSystem.h"
#include "ParticleSystem.cuh"
#include "helper_cuda.h"
#include "helper_math.h"
#include "World.cuh"
#include "aux.h"

__constant__
SysParams system_params;

__global__
void integrate_system(float4 *pos, float4 *vel, float4 *force, unsigned int n_particles)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx>=n_particles) return; 

	float3 vel_f = make_float3(vel[idx] + force[idx]);
	vel_f += system_params.gravity*system_params.dt;
	vel_f *= system_params.global_damping;

	pos[idx] += make_float4(vel_f*system_params.dt) ;
	vel[idx] =  make_float4(vel_f);

	World::checkBoudaries(&pos[idx], &vel[idx]);

}

void ParticleSystem::integrate(){
	unsigned int n_threads, n_blocks;
	computeGridSize(params.n_particles,256, &n_blocks, &n_threads);
	integrate_system<<< n_blocks, n_threads >>>(dPos, dVel, dFor, params.n_particles);
}

void ParticleSystem::copyParticlesToDevice(){
        checkCudaErrors(cudaMemcpy(dPos, hPos, sizeof(float4)*params.n_particles,
        							cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dVel, hVel, sizeof(float4)*params.n_particles,
        							cudaMemcpyHostToDevice));

}

void ParticleSystem::copyParticlesToHost(){
        checkCudaErrors(cudaMemcpy(hPos, dPos, sizeof(float4)*params.n_particles,
        							cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hVel, dVel, sizeof(float4)*params.n_particles,
        							cudaMemcpyDeviceToHost));

}

