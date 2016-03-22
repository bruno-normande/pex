/*
 * SortingAndSearch.h
 *
 *  Created on: 19/03/2016
 *      Author: bruno
 */

#ifndef SORTINGANDSEARCH_H_
#define SORTINGANDSEARCH_H_

#include "ContactDetection.h"
#include <string>

class SortingAndSearch : public ContactDetection {
public:
	SortingAndSearch();
	virtual ~SortingAndSearch();

	void memInitialize();

	void createNeighboorList(float4 *dPos, float4 *dVel);

	void calculateContactForce(float4 *dPos, float4 *dVel, float4 *dFor);

	void setParams(SysParams params);

	inline std::string getName(){return std::string("Sorting Contact Detection");};

protected:

	float4 *dSortedPos;
	float4 *dSortedVel;

	// grid data
	uint4 *dSortedGrid; // grid value for each particle

	//params
	int n_particles;
	float d;
	float3 p_max, p_min;
	int3 gridSize;

protected:
	//mehotds
	void prepareGrid(float4 *dpos);
	void sortParticles();
	void reorderPosAndVel(float4 *dPos, float4 *dVel);

};

#endif /* SORTINGANDSEARCH_H_ */
