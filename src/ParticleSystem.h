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
	ParticleSystem(unsigned int n_particles);
	virtual ~ParticleSystem();

	/** Run simulation */
	void run();

private:
	unsigned int n_particles;
};

#endif /* PARTICLESYSTEM_H_ */
