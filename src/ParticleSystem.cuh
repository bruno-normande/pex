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
						float dt, unsigned int n_particles);

__global__
void simple_sum(float *pos, int N);

#endif /* PARTICLESYSTEM_CUH_ */
