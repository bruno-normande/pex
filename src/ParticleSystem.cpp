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
#include "DirectChecking.h"
#include "SortingContactDetection.h"
#include "CellMapping.h"
#include "World.cuh"

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

ParticleSystem::ParticleSystem(unsigned int n_particles, NeighboorAlg neigh_alg,
								SystemType distr) :
	hPos(NULL), dPos(NULL),
	hVel(NULL), dVel(NULL),
	dFor(NULL)
{
	type = distr; // default

	params.n_particles = n_particles;
	params.particle_radius = 1.0/64.0;
	params.dt = 0.1;
	params.boundary_damping = -0.5;
	params.global_damping = 0.9; // 1.0 what's damping? 0,02
	params.damping = 0.02;
	params.spring = -0.5;
	params.shear = 0.1;
	params.gravity = make_float3(0.0, 0.0, -0.003);
	params.p_max = make_float3(-99.-99,-99.0,0.0);
	params.p_min = make_float3(99.0,99.0,99.0);

	if(type == FLUID){
		params.n_obstacles = 1;
	}else{
		params.n_obstacles = 0;
	}

	switch (neigh_alg) {
		case DM:
			contact = new DirectMapping();
			break;
		case SCD:
			contact = new SortingContactDetection();
			break;
		case CM:
			contact = new CellMapping();
			break;
		default:
			contact = new DirectChecking();
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
	if(hObs)
		delete hObs;
	if(dObs)
		cudaFree(dObs);
}

float ParticleSystem::run(){
	int t_steps = 2500;

	memInitialize();
	createParticles(); //has to come before anny device mem copy
	checkCudaErrors(cudaMemcpyToSymbol(&system_params, &params, sizeof(SysParams)));
	contact->setParams(params);

	contact->memInitialize();
	copyParticlesToDevice();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
        
	dumpXYZ();
	for(int i = 0; i < t_steps; i++){

		integrate();

		// search for neighboors
		contact->createNeighboorList(dPos, dVel);

		// calculate forces
		contact->calculateContactForce(dPos, dVel, dFor);

		if(i%10 == 0){
			copyParticlesToHost();
			dumpXYZ();

		}
		if(i%(t_steps/20)==0){
			std::cout << 100.0*i/t_steps << "% " << std::flush;
		}
	}
	cudaEventRecord(stop);
	dumpXYZ();
	std::cout << std::endl;

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	return milliseconds;
}

void ParticleSystem::memInitialize(){
	// alocate host memory
	// position
	hPos = new float4[params.n_particles];
	hVel = new float4[params.n_particles];
	if(params.n_obstacles)
		hObs = new float4[params.n_obstacles];

	// alocate device memory
	checkCudaErrors(cudaMalloc((void**) &dPos, sizeof(float4) * params.n_particles));
	checkCudaErrors(cudaMalloc((void**) &dVel, sizeof(float4) * params.n_particles));
	checkCudaErrors(cudaMalloc((void**) &dFor, sizeof(float4) * params.n_particles));
	if(params.n_obstacles)
		checkCudaErrors(cudaMalloc((void**) &dObs, sizeof(float4) * params.n_obstacles));

}

void ParticleSystem::createParticles(){
	float jitter = params.particle_radius*0.01;
	unsigned int side = ceilf(powf(params.n_particles, 1.0/3.0));
	unsigned int grid_size[3]; // quantidade de partículas por lado
	float distance = params.particle_radius*2; // distância entre partículas
	float y0 = 0;

	switch(type){

		case SPARSE:
		{
			// igual ao dense mas com Distancia entre partículas maior
			grid_size[0] = grid_size[1] = grid_size[2] = side;
			distance = params.particle_radius*10.0; //TODO: Checar se essa distância está boa
			randomizeVelocity();
			break;
		}

		case FLUID:
		{
			grid_size[0] = side/2;
			grid_size[1] = side/2;
			grid_size[2] = side*4;

			float obstacle_radius = side*params.particle_radius;
			y0 = side*params.particle_radius*2 + params.particle_radius ;

			hObs[0] = make_float4(obstacle_radius);
			break;
		}
		default: // default == dense
		{
			grid_size[0] = grid_size[1] = grid_size[2] = side;
		}

	}

	distributeParticles(grid_size, distance, jitter, y0);
}

void ParticleSystem::distributeParticles(unsigned int* grid_size, float distance,
		float jitter, float y0 = 0)
{

	srand(1);

	for(int z = 0; z < grid_size[2]; z++){
		for(int y = 0; y < grid_size[1]; y++){
			for(int x = 0; x < grid_size[0]; x++){
				unsigned int i = (z*grid_size[1]*grid_size[0]) + (y*grid_size[0]) + x;

				if(i < params.n_particles){
					hPos[i].x = (distance * x) + params.particle_radius + (frand()*2.0-1.0)*jitter;
					if(hPos[i].x > params.p_max.x)
						params.p_max.x = hPos[i].x;
					if(hPos[i].x < params.p_min.x)
						params.p_min.x = hPos[i].x;

					hPos[i].y = (distance * y) + params.particle_radius + (frand()*2.0f-1.0f)*jitter + y0;
					if(hPos[i].y > params.p_max.y)
						params.p_max.y = hPos[i].y;
					if(hPos[i].y < params.p_min.y)
						params.p_min.y = hPos[i].y;

					hPos[i].z = (distance * z) + params.particle_radius + (frand()*2.0f-1.0f)*jitter;
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
	params.p_max.x += distance;
	params.p_max.y += distance;
	params.p_max.z += distance;
	params.p_min.x -= distance;
	params.p_min.y -= distance;
	params.p_min.z -= distance;
	std::cout << "PMAX= (" << params.p_max.x <<", " << params.p_max.y << ", " << params.p_max.z << ")" << std::endl;
	std::cout << "PMAX= (" << params.p_min.x <<", " << params.p_min.y << ", " << params.p_min.z << ")" << std::endl;
}

void ParticleSystem::randomizeVelocity(){
	for(int i = 0; i < params.n_particles; i++){
		hVel[i].x = frand()*2.0 - 1.0;
		hVel[i].y = frand()*2.0 - 1.0;
		hVel[i].z = frand()*2.0 - 1.0;
	}
}

void ParticleSystem::dumpXYZ(){
	if(f_out.is_open()){
		f_out << params.n_particles << std::endl << std::endl;
		for(int i = 0; i < params.n_particles; i++){
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

void ParticleSystem::printInfo(){
	std::cout << "Alg: " << contact->getName() << std::endl;
}











