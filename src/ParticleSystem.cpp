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

#include "helper_cuda.h"
#include "ParticleSystem.h"
#include "ParticleSystem.cuh"
#include "DirectMapping.h"

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

ParticleSystem::ParticleSystem(unsigned int n_particles,
								NeighboorAlg neigh_alg = DM) :
	hPos(NULL), dPos(NULL),
	hVel(NULL), dVel(NULL),
	dFor(NULL)
{
	this->n_particles = n_particles;
	type = DENSE; // default

	params.particle_radius = 1.0/64.0;
	params.dt = 0.01;
	params.boundarie_damping = -0.5;
	params.global_damping = 1.0;
	params.gravity = make_float3(0.0, 0.0, -0.0003);
	params.p_max = make_float3(0.0,0.0,0.0);
	params.p_min = make_float3(0.0,0.0,0.0);

	switch (neigh_alg) {
		case DM:
			contact = new DirectMapping();
			break;
		default:
			break;
	}
}

ParticleSystem::~ParticleSystem() {
	if(hPos)
		delete hPos;
	if(dPos)
		cudaFree(dPos);
	if(hVel)
		delete hVel;
	if(dVel)
		cudaFree(dVel);
	if(dFor)
		cudaFree(dFor);
}

void ParticleSystem::run(){

	memInitialize();
	createParticles();

	copyParticlesToDevice();
	for(int i = 0; i < 100; i++){

		integrate();

		// search for neighboors
		contact->createNeighboorList(dPos);

		// calculate forces

		if(i%5){
			copyParticlesToHost();
			dumpXYZ();
		}
	}
	dumpXYZ();

}

void ParticleSystem::memInitialize(){
	// alocate host memory
	// position
	hPos = new float4[n_particles];
	hVel = new float4[n_particles];

	// alocate device memory
	checkCudaErrors(cudaMalloc((void**) &dPos, sizeof(float4) * n_particles));
	checkCudaErrors(cudaMalloc((void**) &dVel, sizeof(float4) * n_particles));
	checkCudaErrors(cudaMalloc((void**) &dFor, sizeof(float4) * n_particles));
	checkCudaErrors(cudaMemcpyToSymbol(&system_params, &params, sizeof(SysParams)));

	contact->memInitialize();
}

void ParticleSystem::createParticles(){
	float jitter = params.particle_radius*0.01;
	unsigned int side = ceilf(powf(n_particles, 1.0/3.0));
	unsigned int grid_size[3]; // quantidade de partículas por lado
	float distance = params.particle_radius*2; // distância entre partículas

	switch(type){

	case SPARSE:
		// igual ao dense mas com Distancia entre partículas maior
		grid_size[0] = grid_size[1] = grid_size[2] = side;
		distance = params.particle_radius*10.0; //TODO: Checar se essa distância está boa
		randomizeVelocity();
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
					hPos[i].x = (distance * x) + params.particle_radius - 1.0f + (frand()*2.0-1.0)*jitter;
					if(hPos[i].x > params.p_max.x)
						params.p_max.x = hPos[i].x;
					if(hPos[i].x < params.p_min.x)
						params.p_min.x = hPos[i].x;
					hPos[i].y = (distance * y) + params.particle_radius - 1.0f + (frand()*2.0f-1.0f)*jitter;
					if(hPos[i].y > params.p_max.y)
						params.p_max.y = hPos[i].y;
					if(hPos[i].y < params.p_min.y)
						params.p_min.y = hPos[i].y;
					hPos[i].z = (distance * z) + params.particle_radius - 1.0f + (frand()*2.0f-1.0f)*jitter;
					if(hPos[i].z > params.p_max.z)
						params.p_max.z = hPos[i].z;
					if(hPos[i].z < params.p_min.z)
						params.p_min.z = hPos[i].z;

					hPos[i].w = 0;

					hVel[i].x = 0.0;
					hVel[i].y = 0.0;
					hVel[i].z = 0.0;
					hVel[i].w = 0.0;
				}
			}
		}
	}
}

void ParticleSystem::randomizeVelocity(){
	for(int i = 0; i < n_particles; i++){
		hVel[i].x = frand()*2.0 - 1.0;
		hVel[i].y = frand()*2.0 - 1.0;
		hVel[i].z = frand()*2.0 - 1.0;
	}
}

void ParticleSystem::dumpXYZ(){
	if(f_out.is_open()){
		f_out << n_particles << std::endl << std::endl;
		for(int i = 0; i < n_particles; i++){
			f_out << i << " " << hPos[i].x << " " << hPos[i].y << " " << hPos[i].z << std::endl;
		}
	}
}

void ParticleSystem::cleanUp(){
	if(f_out.is_open()){
		f_out.close();
	}
}

void ParticleSystem::setOutputFile(std::string file_name){
	if(f_out.is_open()){
		f_out.close();
	}
	if(!file_name.empty())
		f_out.open(file_name.c_str());
}











