/*
 * ContactDetection.h
 *
 *  Created on: 01/02/2016
 *      Author: bruno
 */

#ifndef CONTACTDETECTION_H_
#define CONTACTDETECTION_H_

#include <cstdlib>
#include <cstdio>
#include <string>
#include <thrust/device_vector.h>

#include "helper_math.h"
#include "aux.h"

class ContactDetection {
public:
	ContactDetection(){}
	virtual ~ContactDetection(){}

	virtual void memInitialize() = 0;

	virtual void createNeighboorList(thrust::host_vector<float4>& dPos, thrust::host_vector<float4> & dVel) = 0;

	virtual void calculateContactForce(thrust::host_vector<float4>& dPos, thrust::host_vector<float4>& dVel, 
										thrust::host_vector<float4>& dFor) = 0;

	virtual void setParams(SysParams params) = 0;

	virtual std::string getName() = 0;
};

#endif /* CONTACTDETECTION_H_ */
