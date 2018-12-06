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

	void createNeighboorList(thrust::device_vector<float4>& dPos, 
							thrust::device_vector<float4> & dVel, unsigned int n_particles){}

	// Colocar para receber ponteiro dFor
	void calculateContactForce(thrust::device_vector<float4>& dPos, thrust::device_vector<float4>& dVel, 
										thrust::device_vector<float4>& dFor, unsigned int n_particles);

	inline void setParams(SysParams params){n_particles = params.n_particles;}

	inline std::string getName(){return std::string("Direct Checking");}
protected:
	unsigned int n_particles;
};

#endif /* DIRECTCHECKING_H_ */
