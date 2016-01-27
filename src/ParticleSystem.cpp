/*
 * ParticleSystem.cpp
 *
 *  Created on: 24/01/2016
 *      Author: bruno
 */

#include <cuda_runtime.h>
#include <cuda.h>

#include "ParticleSystem.h"
#include "helper_cuda.h"

ParticleSystem::ParticleSystem(unsigned int n_particles) :
	hPos(NULL),
	dPos(NULL)
{
	this->n_particles = n_particles;

}

ParticleSystem::~ParticleSystem() {
	if(hPos)
		delete hPos;

	if(dPos)
		cudaFree(dPos);
}

void ParticleSystem::run(){

}

void ParticleSystem::memInitialize(){
	// alocate host memory
	// position
	hPos = new float[n_particles*4]; //TODO: pq * 4?

	// alocate device memory
	checkCudaErrors(cudaMalloc((void**) dPos, sizeof(float) * n_particles * 4));


}
