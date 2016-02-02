/*
 * ContactDetection.h
 *
 *  Created on: 01/02/2016
 *      Author: bruno
 */

#ifndef CONTACTDETECTION_H_
#define CONTACTDETECTION_H_

class ContactDetection {
public:
	ContactDetection(){}
	virtual ~ContactDetection(){}

	virtual void memInitialize() = 0;

	virtual void createNeighboorList(float4 *dPos) = 0;

	// Colocar para receber ponteiro dFor
	virtual void calculateContactForce() = 0;
	virtual void setPMin(float3 pMin) = 0;
	virtual void setPMax(float3 pMax) = 0;
};

#endif /* CONTACTDETECTION_H_ */
