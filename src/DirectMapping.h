/*
 * DirectMapping.h
 *
 *  Created on: 01/02/2016
 *      Author: bruno
 */

#ifndef DIRECTMAPPING_H_
#define DIRECTMAPPING_H_

#include "ContactDetection.h"
#include "ParticleSystem.h"

class DirectMapping : public ContactDetection {
public:
	DirectMapping();
	virtual ~DirectMapping();

	void memInitialize();
	void createNeighboorList(float4 *dPos);
	void calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor);
	std::string getName();
	void setParams(SysParams params);

protected:
	int3 gridDim;
	unsigned int n_particles;
	float d;

	int *dGrid; // stores lists heads
	int *dList; // stores grid's lists
};

#endif /* DIRECTMAPPING_H_ */
