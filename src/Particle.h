/*
 * Particle.h
 *
 *  Created on: 27/01/2016
 *      Author: bruno
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

class Particle {
public:
	__device__ __host__ 
	Particle(){};
	__device__ __host__ 
	Particle(unsigned int id, float* pos, float* vel);
	__device__ __host__ 
	~Particle();

private:
	unsigned int id;
	float pos[3];
	float vel[3];

};

#endif /* PARTICLE_H_ */
