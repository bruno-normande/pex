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

class CellMapping : public ContactDetection {
public:
	CellMapping();
	virtual ~CellMapping();

	void memInitialize();
	void createNeighboorList(float4 *dPos, float4 *dVel);
	void calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor);
	std::string getName();
	void setParams(SysParams params);

protected:
	float d;
	unsigned int n_particles;
	int3 gridDim;
	float3 p_min;

	int *dGrid; // grid of lists
	int *dGridCounter; // grid of lists
};

#endif /* DIRECTMAPPING_H_ */
