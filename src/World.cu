/*
 * World.cpp
 *
 *  Created on: 29/01/2016
 *      Author: bruno
 */

#include "World.cuh"

float World::boundarie_damping = -5.0;

__device__ __host__
void World::checkBoudaries(float4* pos, float4* vel,
		const float &radius)
{
	// inicialmente vamos apenas impedir as particulas de passarem
	// pelo chão
	if(pos->y < -1.0){
		pos->y = -1.0 + radius;
		vel->y *= boundarie_damping;
	}
}

