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

	void memInitialize() = 0;

	void createNeighboorList(float4 *dPos) = 0;

	// Colocar para receber ponteiro dFor
	void calculateContactForce() = 0;
};

#endif /* CONTACTDETECTION_H_ */
