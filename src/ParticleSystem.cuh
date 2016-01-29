/*
 * ParticleSystem.cuh
 *
 *  Created on: 27/01/2016
 *      Author: bruno
 */

#ifndef PARTICLESYSTEM_CUH_
#define PARTICLESYSTEM_CUH_

__global__
void integrate_system(float *pos, float *vel,
						float dt, unsigned int n_particles)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index >= n_particles) return;

	pos[index*4] = pos[index*4 + 1] = pos[index*4 + 2] = pos[index*4 + 3] = index;

}

#endif /* PARTICLESYSTEM_CUH_ */
