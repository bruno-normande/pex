
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
void integrate_system(float4 *pos, float4 *vel, float4 *force, float4 *obstacles,
		unsigned int n_particles, unsigned int n_obstacles)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx>=n_particles) return; 
	float3 my_pos = make_float3(pos[idx]);
	float3 my_vel = make_float3(vel[idx]);

	float3 my_force = make_float3(force[idx]);
	for(int i = 0; i < n_obstacles; i++){
		my_force += World::contactForce(my_pos, make_float3(obstacles[i]),
				my_vel, make_float3(0), system_params.particle_radius,
				obstacles[i].z);
	}

	float3 vel_f = my_vel + my_force;
	vel_f += system_params.gravity*system_params.dt;
	vel_f *= system_params.global_damping;

	pos[idx] += make_float4(vel_f*system_params.dt) ;
	vel[idx] =  make_float4(vel_f);

	World::checkBoudaries(&pos[idx], &vel[idx]);

}

void ParticleSystem::integrate(){
	unsigned int n_threads, n_blocks;
	computeGridSize(params.n_particles,256, &n_blocks, &n_threads);
	integrate_system<<< n_blocks, n_threads >>>(dPos, dVel, dFor, dObs, params.n_particles, params.n_obstacles);
}

void ParticleSystem::copyParticlesToDevice(){
        checkCudaErrors(cudaMemcpy(dPos, hPos, sizeof(float4)*params.n_particles,
        							cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dVel, hVel, sizeof(float4)*params.n_particles,
        							cudaMemcpyHostToDevice));
	if(params.n_obstacles)
        	checkCudaErrors(cudaMemcpy(dObs, hObs, sizeof(float4)*params.n_obstacles,
                							cudaMemcpyHostToDevice));

}

void ParticleSystem::copyParticlesToHost(){
        checkCudaErrors(cudaMemcpy(hPos, dPos, sizeof(float4)*params.n_particles,
        							cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hVel, dVel, sizeof(float4)*params.n_particles,
        							cudaMemcpyDeviceToHost));

}

