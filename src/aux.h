/*
 * aux.h
 *
 *  Created on: 02/02/2016
 *      Author: bruno
 */

#ifndef AUX_H_
#define AUX_H_

#include <math.h>

inline void computeGridSize(unsigned int n, unsigned int block_size,
						unsigned int *n_blocks, unsigned int *n_threads)
{
	*n_threads = std::min(block_size, n);
	*n_blocks = ceil((float)n/(float)(*n_threads));
}

__device__
inline int3 get_grid_pos(float3 pos, float3 p_min, float d){
	int3 gridPos;
	gridPos.x = floor( (pos.x - p_min.x) / d );
	gridPos.y = floor( (pos.y - p_min.y) / d );
	gridPos.z = floor( (pos.z - p_min.z) / d );
	return gridPos;
}

#endif /* AUX_H_ */
