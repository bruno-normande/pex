/*
 * ParticleSystem.cuh
 *
 *  Created on: 27/01/2016
 *      Author: bruno
 */

#ifndef PARTICLESYSTEM_CUH_
#define PARTICLESYSTEM_CUH_

__global__
void integrate_system(float4 *pos, float4 *vel,
						float dt, unsigned int n_particles,
						float damping);

#endif /* PARTICLESYSTEM_CUH_ */
