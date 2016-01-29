

#include "ParticleSystem.h"
#include "ParticleSystem.cuh"


__global__
void integrate_system(float *pos, float *vel,
						float dt, unsigned int n_particles)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index >= n_particles) return;

	pos[index*4] = pos[index*4 + 1] = pos[index*4 + 2] = pos[index*4 + 3] = index;

}

void ParticleSystem::integrate(){
	unsigned int n_threads, n_blocks;
	computeGridSize(n_particles,256, &n_blocks, &n_threads);

	integrate_system<<< n_blocks, n_threads >>>(dPos, dVel, dt, n_particles);
}
