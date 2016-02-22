/*
 * ParticleSystem.h
 *
 *  Created on: 24/01/2016
 *      Author: bruno
 */

#ifndef PARTICLESYSTEM_H_
#define PARTICLESYSTEM_H_

#include <algorithm>
#include <string>
#include <fstream>
#include "aux.h"
#include "ContactDetection.h"

class ParticleSystem {
public:
	ParticleSystem(unsigned int n_particles, NeighboorAlg neigh_alg = DM, SystemType distr = DENSE);
	virtual ~ParticleSystem();

	/** Run simulation */
	float run();
	/** Initialize system memory */
	void memInitialize();

	/** Initialize particles */
	void createParticles();

	/** Distribute particles in cube */
	void distributeParticles(unsigned int* grid_size, float distance, float jitter, float y0);

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

	void printInfo();

protected:
	float4 *hPos;
	float4 *dPos;
	float4 *hVel;
	float4 *dVel;
	float4 *dFor;
	float4 *hObs;
	float4 *dObs;

	SysParams params;

	SystemType type;

	std::ofstream f_out;

	ContactDetection *contact;
};

#endif /* PARTICLESYSTEM_H_ */
