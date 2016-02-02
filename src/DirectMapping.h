/*
 * DirectMapping.h
 *
 *  Created on: 01/02/2016
 *      Author: bruno
 */

#ifndef DIRECTMAPPING_H_
#define DIRECTMAPPING_H_

#include "ContactDetection.h"

class DirectMapping : ContactDetection {
public:
	DirectMapping(float3 pMax, float3 pMin, float d, unsigned int n_particle);
	virtual ~DirectMapping();

	void memInitialize();
	void createNeighboorList(float4 *dPos);
	void calculateContactForce();

protected:
	int3 gridDim;
	unsigned int n_particle;

	int *dGrid; // stores lists heads
	int *dList; // stores grid's lists
};

#endif /* DIRECTMAPPING_H_ */
