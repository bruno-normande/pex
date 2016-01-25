/*
 * ParticleSystem.h
 *
 *  Created on: 24/01/2016
 *      Author: bruno
 */

#ifndef PARTICLESYSTEM_H_
#define PARTICLESYSTEM_H_

#define DENSE = 0

class ParticleSystem {
public:
	ParticleSystem(uint n_particles);
	virtual ~ParticleSystem();

	/** Run simulation */
	void run();

private:
	uint n_particles;
};

#endif /* PARTICLESYSTEM_H_ */
