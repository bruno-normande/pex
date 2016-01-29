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
	static void checkBoudaries(float4* pos, float4* vel, const float &radius);

private:
	static float boundarie_damping;
};

#endif /* WORLD_H_ */
