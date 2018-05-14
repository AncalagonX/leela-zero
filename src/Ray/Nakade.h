#ifndef _NAKADE_H_
#define _NAKADE_H_

#include "GoBoard.h"

////////////
//    //
////////////

const int NOT_NAKADE = -1;

const int NAKADE_QUEUE_SIZE = 30;

//////////////
//    //
//////////////
struct nakade_queue_t {
  int pos[NAKADE_QUEUE_SIZE];
  int head, tail;
};

////////////
//    //
////////////
// 
void InitializeNakadeHash( void );

// ()
bool IsNakadeSelfAtari( const game_info_t *game, const int pos, const int color );

// (UCT)
bool IsUctNakadeSelfAtari( const game_info_t *game, const int pos, const int color );

// 
// , 
// , -1
void SearchNakade( const game_info_t *game, int *nakade_num, int *nakade_pos );

// 
// , 
// , -1
int CheckRemovedStoneNakade( const game_info_t *game, const int color );

#endif
