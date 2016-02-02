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
#include <string>
#include <fstream>

#include "ContactDetection.h"

enum SystemType { DENSE, SPARSE, FLUID };
enum NeighboorAlg { DM };

struct SysParams{
	float3 gravity;
	float dt;

	float particle_radius;
	float3 p_max; // Max x, y, z
	float3 pmin; // Min x, y, z

	float boundarie_damping;
	float global_damping;

};

class ParticleSystem {
public:
	ParticleSystem(unsigned int n_particles, NeighboorAlg neigh_alg = DM);
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

	/** Clean everything up after finish */
	void cleanUp();

	/** Set output file */
	void setOutputFile(std::string file_name);

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
	float4 *dFor;

	SysParams params;

	SystemType type;

	unsigned int n_particles;

	std::ofstream f_out;

	ContactDetection *contact;
};

#endif /* PARTICLESYSTEM_H_ */
