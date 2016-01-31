/**
 * Created by Bruno
 * 2016
 */

#include <cuda_runtime.h>
#include <boost/program_options.hpp>
#include <string>
#include <iostream>

#include "ParticleSystem.h"

#define GRID_SIZE       64
#define NUM_PARTICLES   5

int main(int argc, char **argv) {
	unsigned int n_particles = NUM_PARTICLES;

	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()
			("help,h", "Produces help message")
			("file,f", po::value<std::string>(), "Output file")
			(",n", po::value<unsigned int>(), "Output file");

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,desc), vm);
	po::notify(vm);

	if(vm.coun("help")){
		std::cout << desc << std::endl;
		return 0;
	}

	if(vm.count("n")){
		n_particles = vm["n"].as<unsigned int>();
	}
	ParticleSystem *system = new ParticleSystem(n_particles);

	std::cout << "Starting simulation..." << std::endl << "N = " << n_particles;
	std::string out_file;
	if(vm.count("file")){
		out_file = vm["file"].as<std::string>();
		system->setOutputFile(out_file);
		std::cout << "Output = " << out_file << std::endl;
	}
	system->run();
	system->cleanUp();

	//TODO Fazer grid-size relatico a quantidade de partículas e a
	// 		tipo de simulação por linha de comando

	//TODO botar para selecionar device por entrada tbm


	// cleanup
	delete system;
	cudaDeviceReset();
}




