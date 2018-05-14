#ifndef _LADDER_H_
#define _LADDER_H_

#include "GoBoard.h"

// 
void LadderExtension( game_info_t *game, int color, bool *ladder_pos );
// 
bool CheckLadderExtension( game_info_t *game, int color, int pos );
#endif
