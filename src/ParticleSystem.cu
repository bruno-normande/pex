
#include <iostream>

__device__
void helloSphere(){
	unsigned int pos_x = blockIdx.x;
	unsigned int dim_x = blockDim.x;
	unsigned int thread_x = threadIdx.x;

	std::cout << "Hi from " << pos_x << " - " << dim_x << " - " << thread_x << std::endl;
}
