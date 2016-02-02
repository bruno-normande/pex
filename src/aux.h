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



#endif /* AUX_H_ */
