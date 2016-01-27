/*
 * Particle.cpp
 *
 *  Created on: 27/01/2016
 *      Author: bruno
 */

#include "Particle.h"

__device__ __host__ Particle(unsigned int id, float* pos, float* vel){
	this->id = id;
	this->pos[0] = pos[0];
	this->pos[1] = pos[1];
	this->pos[2] = pos[2];
	this->vel[0] = vel[0];
	this->vel[1] = vel[1];
	this->vel[2] = vel[2];
}

__device__ __host__ Particle::~Particle(){
	// TODO Auto-generated destructor stub
}

