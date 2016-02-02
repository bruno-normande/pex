/*
 * DirectChecking.h
 *
 *  Created on: 02/02/2016
 *      Author: bruno
 */

#ifndef DIRECTCHECKING_H_
#define DIRECTCHECKING_H_

#include "ContactDetection.h"

class DirectChecking : public ContactDetection {
public:
	DirectChecking(unsigned int n_particles):n_particles(n_particles){}
	virtual ~DirectChecking(){}

	void memInitialize(){}

	void createNeighboorList(float4 *dPos){}

	// Colocar para receber ponteiro dFor
	void calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor);

	void setMinMax(float3 pMin, float3 pMax){}
protected:
	unsigned int n_particles;
};

#endif /* DIRECTCHECKING_H_ */
