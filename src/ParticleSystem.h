/*
 * ParticleSystem.h
 *
 *  Created on: 24/01/2016
 *      Author: bruno
 */

#ifndef PARTICLESYSTEM_H_
#define PARTICLESYSTEM_H_

#include <algorithm>
#include <math.h>

struct SysParams{
	float3 gravity;
	float dt;

	float particle_radius;

	float boundarie_damping;
	float global_damping;
};

enum SystemType { DENSE, SPARSE, FLUID };

class ParticleSystem {
public:
	ParticleSystem(unsigned int n_particles);
	virtual ~ParticleSystem();

	/** Run simulation */
	void run();
	/** Initialize system memory */
	void memInitialize();

	/** Initialize particles */
	void createParticles();

	/** Distribute particles in cube */
	void distributeParticles(unsigned int* grid_size, float distance, float jitter);

	/** Distribute random velocity to all particles */
	void randomizeVelocity();

	/** */
	void copyParticlesToDevice();
	/** */
	void copyParticlesToHost();

	/** Dump to stdout particles location */
	void dumpXYZ();

	/** Integrate system */
	void integrate();

	inline void computeGridSize(unsigned int n, unsigned int block_size,
						unsigned int *n_blocks, unsigned int *n_threads)
	{
		*n_threads = std::min(block_size, n);
		*n_blocks = ceil((float)n/(float)(*n_threads));
	}

protected:
	float4 *hPos;
	float4 *dPos;
	float4 *hVel;
	float4 *dVel;

	SysParams params;

	SystemType type;

	unsigned int n_particles;
};

#endif /* PARTICLESYSTEM_H_ */
