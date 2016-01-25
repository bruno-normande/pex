

#include <iostream>

#include <cuda_runtime.h>

#include "ParticleSystem.h"

#define GRID_SIZE       64
#define NUM_PARTICLES   16384

int main(int argc, char **argv) {
	std::cout << "Iniciando... " << std::endl;

	uint n_particles = NUM_PARTICLES;
	uint grid_size = GRID_SIZE;

	//TODO: adicionar número de partículas por linha de comando

	//TODO Fazer grid-size relatico a quantidade de partículas e a
	// 		tipo de simulação

	//TODO botar para selecionar device por entrada tbm

	ParticleSystem *system = ParticleSystem(n_particles);

	system->run();

	// cleanup
	delete system;
	cudaDeviceReset();
}




