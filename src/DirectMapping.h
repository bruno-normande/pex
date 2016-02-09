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

// máximo de partículas por célula
#define CELL_MAX_P 8
#define EMPTY -1

class DirectMapping : public ContactDetection {
public:
	DirectMapping();
	virtual ~DirectMapping();

	void memInitialize();
	void createNeighboorList(float4 *dPos, float4 *dVel);
	void calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor);
	std::string getName();
	void setParams(SysParams params);

protected:
	int3 gridDim;
	unsigned int n_particles;
	float d;
	float3 p_min;

	int *dGrid; // grid of lists
};

#endif /* DIRECTMAPPING_H_ */
