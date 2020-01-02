//
// Created by francesco on 12/23/19.
//

#ifndef LPTML_TYPES_H
#define LPTML_TYPES_H

#include <utility>

#define float_type float
#define unsigned_type unsigned int

#define row_type std::vector<float_type>
#define matrix_type std::vector<row_type>
#define label_row_type std::vector<unsigned_type>

#define pair_type std::vector<std::pair<std::vector<float_type>, std::vector<float_type>>>
#define pair_index_type std::vector<std::pair<unsigned_type, unsigned_type>>

float_type as_float_type(float el){
    return *(float_type *) (&el);
}

unsigned_type as_unsigned_type(unsigned int el){
    return *(unsigned_type *) (&el);
}

#endif //LPTML_TYPES_H
