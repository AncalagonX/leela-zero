#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <chrono>

////////////
//    //
////////////
typedef std::chrono::high_resolution_clock ray_clock;

// 
inline double GetSpendTime(const ray_clock::time_point& start_time) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(ray_clock::now() - start_time).count() / 1000.0;
}

//  (float)
void InputTxtFLT( const char *filename, float *ap, const int array_size );

//  (double)
void InputTxtDBL( const char *filename, double *ap, const int array_size );

#endif
