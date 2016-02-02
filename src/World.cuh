/*
 * World.h
 *
 *  Created on: 29/01/2016
 *      Author: bruno
 */

#ifndef WORLD_H_
#define WORLD_H_

#include "helper_math.h"

class World {
public:

	/** Atualiza posição da partícula para que ela não
	 * atravesse nenhum objeto ou parede */
	__device__
	static void checkBoudaries(float4* pos, float4* vel);

	/** Calculates the resulting force from two particles
	 * contact
	 */
	__device__
	static float3 contactForce(float4 posA, float4 posB,
			float3 velA, float3 velB,
	        float radiusA, float radiusB);

};

#endif /* WORLD_H_ */
