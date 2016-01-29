/*
 * ParticleSystem.cpp
 *
 *  Created on: 24/01/2016
 *      Author: bruno
 */

#include <math.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include "ParticleSystem.h"
#include "helper_cuda.h"

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

ParticleSystem::ParticleSystem(unsigned int n_particles) :
	hPos(NULL), dPos(NULL),
	hVel(NULL), dVel(NULL)
{
	this->n_particles = n_particles;
	particle_radius = DEFAULT_RADIUS;
	type = DENSE; // default
}

ParticleSystem::~ParticleSystem() {
	if(hPos)
		delete hPos;

	if(dPos)
		cudaFree(dPos);
}

void ParticleSystem::run(){

	memInitialize();
	createParticles();
	copyParticlesToDevice();

	dumpXYZ();

}

void ParticleSystem::memInitialize(){
	// alocate host memory
	// position
	hPos = new float[n_particles*POS_DIM];
	hVel = new float[n_particles*VEL_DIM];

	// alocate device memory
	checkCudaErrors(cudaMalloc((void**) dPos, sizeof(float) * n_particles * POS_DIM));
	checkCudaErrors(cudaMalloc((void**) dVel, sizeof(float) * n_particles * VEL_DIM));
}

void ParticleSystem::createParticles(){
	float jitter = particle_radius*0.01;
	unsigned int side = ceilf(powf(n_particles, 1.0/3.0));
	unsigned int grid_size[3]; // quantidade de partículas por lado
	float distance = particle_radius*2; // distância entre partículas

	switch(type){

	case SPARSE:
		// igual ao dense mas com Distancia entre partículas maior
		grid_size[0] = grid_size[1] = grid_size[2] = side;
		distance = particle_radius*10.0; //TODO: Colocar velocidades alleatórias depois
		break;

	case FLUID:
		grid_size[0] = side/2;
		grid_size[1] = side/2;
		grid_size[2] = side*4;
		break;

	default: // default == dense
		grid_size[0] = grid_size[1] = grid_size[2] = side;

	}

	distributeParticles(grid_size, distance, jitter);
}

void ParticleSystem::distributeParticles(unsigned int* grid_size, float distance, float jitter){

	srand(1);

	for(int z = 0; z < grid_size[2]; z++){
		for(int y = 0; y < grid_size[1]; y++){
			for(int x = 0; x < grid_size[0]; x++){
				unsigned int i = (z*grid_size[1]*grid_size[0]) + (y*grid_size[0]) + x;

				if(i < n_particles){
					hPos[i*4] = (distance * x) + particle_radius - 1.0f + (frand()*2.0-1.0)*jitter;
					hPos[i*4+1] = (distance * y) + particle_radius - 1.0f + (frand()*2.0f-1.0f)*jitter;
					hPos[i*4+2] = (distance * z) + particle_radius - 1.0f + (frand()*2.0f-1.0f)*jitter;
					hPos[i*4+2] = 1.0;

					hVel[i*4] = 0.0;
					hVel[i*4+1] = 0.0;
					hVel[i*4+2] = 0.0;
					hVel[i*4+3] = 0.0;
				}
			}
		}
	}
}

void ParticleSystem::copyParticlesToDevice(){
	checkCudaErrors(cudaMemcpy((void*)dPos, (void*)hPos, sizeof(float)*POS_DIM*n_particles, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy((void*)dVel, (void*)hVel, sizeof(float)*VEL_DIM*n_particles, cudaMemcpyHostToDevice));

}

void ParticleSystem::dumpXYZ(){
	std::cout << n_particles << std::endl << std::endl;
	for(int i = 0; i < n_particles; i++){
		std::cout << i << " " << hPos[0] << " " << hPos[1] << " " << hPos[2] << std::endl;
	}
}











