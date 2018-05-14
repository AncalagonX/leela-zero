#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#include "GoBoard.h"
#include "UctSearch.h"


//  
void SetDebugMessageMode( const bool flag );

//  
void PrintBoard( const game_info_t *game );

//                
//    ,           
//    ,   
//    ID
void PrintString( const game_info_t *game );

//  ID  
void PrintStringID( const game_info_t *game );

//  (Debug)
void PrintStringNext( const game_info_t *game );

//   
void PrintLegal( const game_info_t *game, const int color );

//  
void PrintOwner( const uct_node_t *root, const int color, double *own );

//  
void PrintBestSequence( const game_info_t *game, const uct_node_t *uct_node, const int root, const int start_color );

//  
void PrintPlayoutInformation( const uct_node_t *root, const po_info_t *po_info, const double finish_time, const int pre_simulated );

//  
void PrintPoint( const int pos );

//  
void PrintKomiValue( void );

//  Pondering
void PrintPonderingCount( const int count );

//  
void PrintPlayoutLimits( const double time_limit, const int playout_limit );

//  
void PrintReuseCount( const int count );

#endif
