/*
 * SortingContactDetection.h
 *
 *  Created on: 04/02/2016
 *      Author: bruno
 */

#ifndef SORTINGCONTACTDETECTION_H_
#define SORTINGCONTACTDETECTION_H_

#include "ContactDetection.h"
#include <string>

class SortingContactDetection : public ContactDetection {
public:
	SortingContactDetection();
	virtual ~SortingContactDetection();

	void memInitialize();

	void createNeighboorList(float4 *dPos, float4 *dVel);

	void calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor);

	void setParams(SysParams params);

	inline std::string getName(){return std::string("Sorting Contact Detection");};

protected:
	// no original ele guarda ParticleHash, CellStart e CellEnd tbm no host

	float4 *dSortedPos;
	float4 *dSortedVel;

	// grid data
	unsigned int *dGridParticleHash; // grid hash value for each particle
	unsigned int *dGridParticleIndex;// particle index for each particle
	unsigned int *dCellStart;        // index of start of each cell in sorted list
	unsigned int *dCellEnd;          // index of end of cell

	unsigned int    gridSortBits;

	//params
	int n_particles;
	float d;
	float3 p_max, p_min;
	int3 gridSize;

protected:
	//mehotds
	void calcHash(float4 *dpos);
	void sortParticles();
	void reorderAndSetStart(float4 *dPos, float4 *dVel);

};

#endif /* SORTINGCONTACTDETECTION_H_ */
