/*
 * DirectChecking.h
 *
 *  Created on: 02/02/2016
 *      Author: bruno
 */

#ifndef DIRECTCHECKING_H_
#define DIRECTCHECKING_H_

#include <thrust/device_vector.h>

#include "ContactDetection.h"
#include "aux.h"

class DirectChecking : public ContactDetection {
public:
	DirectChecking():n_particles(0){}
	virtual ~DirectChecking(){}

	void memInitialize(){}

	void createNeighboorList(thrust::host_vector<float4>& dPos, thrust::host_vector<float4> & dVel){}

	// Colocar para receber ponteiro dFor
	void calculateContactForce(thrust::host_vector<float4>& dPos, thrust::host_vector<float4>& dVel, 
										thrust::host_vector<float4>& dFor);

	inline void setParams(SysParams params){n_particles = params.n_particles;}

	inline std::string getName(){return std::string("Direct Checking");}
protected:
	unsigned int n_particles;
};

#endif /* DIRECTCHECKING_H_ */
