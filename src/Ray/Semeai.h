#ifndef _SEMEAI_H_
#define _SEMEAI_H_

#include "GoBoard.h"

enum LIBERTY_STATE {
  L_DECREASE,
  L_EVEN,
  L_INCREASE,
};


//  1
bool IsCapturableAtari( const game_info_t *game, const int pos, const int color, const int opponent_pos );

//  
int CheckOiotoshi( const game_info_t *game, const int pos, const int color, const int opponent_pos );

//  
int CapturableCandidate( const game_info_t *game, const int id );

//    
bool IsDeadlyExtension( const game_info_t *game, const int color, const int id );

//  
int CheckLibertyState( const game_info_t *game, const int pos, const int color, const int id );

//  
bool IsSelfAtariCapture( const game_info_t *game, const int pos, const int color, const int id );

//  1()
bool IsCapturableAtariForSimulation( const game_info_t *game, const int pos, const int color, const int id );

//  
bool IsSelfAtariCaptureForSimulation( const game_info_t *game, const int pos, const int color, const int lib );

//  
bool IsSelfAtari( const game_info_t *game, const int color, const int pos );

//  
bool IsAlreadyCaptured( const game_info_t *game, const int color, const int id, int player_id[], int player_ids );

#endif
