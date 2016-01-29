/*
 * ParticleSystem.h
 *
 *  Created on: 24/01/2016
 *      Author: bruno
 */

#ifndef PARTICLESYSTEM_H_
#define PARTICLESYSTEM_H_

#define DEFAULT_RADIUS 1.0/64
#define POS_DIM 4 //TODO PQ 4?!?!?!
#define VEL_DIM 4

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
	void distributeParticles(int* grid_size, float distance, float jitter);

	/** */
	void copyParticlesToDevice();

protected:
	float *hPos;
	float *dPos;
	float *hVel;
	float *dVel;
	float particle_radius;

	SystemType type;

	unsigned int n_particles;
};

#endif /* PARTICLESYSTEM_H_ */
