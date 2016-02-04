/*
 * World.cpp
 *
 *  Created on: 29/01/2016
 *      Author: bruno
 */

#include "World.cuh"
#include "ParticleSystem.cuh"
#include "helper_math.h"
#include "ParticleSystem.h"

extern __constant__
SysParams system_params;

__device__
void World::checkBoudaries(float4* pos, float4* vel)
{
	// impede as partículas de passarem pelas bordas
	if (pos.x > system_params.p_max.x - params.particleRadius)
	{
		pos.x = system_params.p_max.x - params.particleRadius;
		vel.x *= params.boundaryDamping;
	}

	if (pos.x < system_params.p_min.x + params.particleRadius)
	{
		pos.x = system_params.p_min.x + params.particleRadius;
		vel.x *= params.boundaryDamping;
	}

	if (pos.y > system_params.p_max.y - params.particleRadius)
	{
		pos.y = system_params.p_max.y - params.particleRadius;
		vel.y *= params.boundaryDamping;
	}
	if (pos.y < system_params.p_min.y + params.particleRadius)
	{
		pos.y = system_params.p_min.y + params.particleRadius;
		vel.y *= params.boundaryDamping;
	}

	if (pos.z > system_params.p_max.z - params.particleRadius)
	{
		pos.z = system_params.p_max.z - params.particleRadius;
		vel.z *= params.boundaryDamping;
	}

	if (pos.z < system_params.p_min.z + params.particleRadius)
	{
		pos.z = system_params.p_min.z + params.particleRadius;
		vel.z *= params.boundaryDamping;
	}

}

__device__
float3 World::contactForce(float3 posA, float3 posB,
		float3 velA, float3 velB,
        float radiusA, float radiusB)
{
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0);

	if(dist < collideDist){
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel = velB - velA;

		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force // spring = 0.5
		force = -0.5*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += system_params.global_damping*relVel;
		// tangential shear force //shear = 0.1
		force += 0.1*tanVel;

	}

	return force;
}

