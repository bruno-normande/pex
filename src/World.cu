/*
 * World.cpp
 *
 *  Created on: 29/01/2016
 *      Author: bruno
 */

#include "World.cuh"
#include "ParticleSystem.cuh"

__device__
void World::checkBoudaries(float4* pos, float4* vel)
{
	// inicialmente vamos apenas impedir as particulas de passarem
	// pelo chão
	if(pos->z < -1.0){
		pos->z = -1.0 + system_params.particle_radius;
		vel->z *= system_params.boundarie_damping;
	}
}

