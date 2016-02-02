/*
 * DirectMapping.h
 *
 *  Created on: 01/02/2016
 *      Author: bruno
 */

#ifndef DIRECTMAPPING_H_
#define DIRECTMAPPING_H_

#include "ContactDetection.h"

class DirectMapping : public ContactDetection {
public:
	DirectMapping( unsigned int n_particle);
	virtual ~DirectMapping();

	void memInitialize();
	void createNeighboorList(float4 *dPos);
	void calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor);

	void setMinMax(float3 pMin, float3 pMax);

protected:
	int3 gridDim;
	unsigned int n_particle;

	int *dGrid; // stores lists heads
	int *dList; // stores grid's lists
};

#endif /* DIRECTMAPPING_H_ */
