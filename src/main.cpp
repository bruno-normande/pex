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
			(",n", po::value<unsigned int>(), "Particle number")
			("algorithm,a", po::value<std::string>(), "Contact Detection algorithm");

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,desc), vm);
	po::notify(vm);

	if(vm.count("help")){
		std::cout << desc << std::endl;
		return 0;
	}

	if(vm.count("-n")){
		//std::cout << vm << std::endl;
		n_particles = vm["-n"].as<unsigned int>();
	}
	NeighboorAlg neighAlg = DM;
        if(vm.count("algorithm")){
                std::string alg = vm["algorithm"].as<std::string>();
                if(alg=="DC") neighAlg = DC;
                if(alg=="SCD") neighAlg = SCD;
        }


	ParticleSystem *system = new ParticleSystem(n_particles, neighAlg);

	std::cout << "Starting simulation..." << std::endl << "N = " << n_particles << std::endl;
	std::string out_file;
	if(vm.count("file")){
		out_file = vm["file"].as<std::string>();
		system->setOutputFile(out_file);
		std::cout << "Output = " << out_file << std::endl;
	}

	float total_time;

	system->printInfo();
	total_time = system->run();
	system->cleanUp();

	std::cout << "Total time = " << total_time << " ms" << std::endl;

	//TODO Fazer grid-size relatico a quantidade de partículas e a
	// 		tipo de simulação por linha de comando

	//TODO botar para selecionar device por entrada tbm


	// cleanup
	delete system;
	cudaDeviceReset();
}




