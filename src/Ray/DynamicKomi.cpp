#include <iomanip>
#include <iostream>

#include "DynamicKomi.h"
#include "GoBoard.h"
#include "Message.h"

using namespace std;


////////////
//    //
////////////

// 
static int handicap_num = 0;

// 
static int const_handicap_num = 0;

// Dynamic Komi
enum DYNAMIC_KOMI_MODE dk_mode = DK_OFF;


////////////
//    //
////////////

// Dynamic Komi
static void LinearHandicap( const game_info_t *game );

// Dynamic Komi
static void ValueSituational( const uct_node_t *root, const int color );


////////////////////////
//    //
////////////////////////
void
SetConstHandicapNum( const int num )
{
  const_handicap_num = num;
}


////////////////////////
//    //
////////////////////////
void
SetHandicapNum( const int num )
{
  if (const_handicap_num == 0) {
    handicap_num = num;
    if (dk_mode != DK_OFF && 
	handicap_num == 0) {
      dk_mode = DK_OFF;
    } else if (dk_mode == DK_OFF &&
	       handicap_num != 0) {
      dk_mode = DK_LINEAR;
    } 
  } else {
    handicap_num = const_handicap_num;
    dk_mode = DK_LINEAR;
  }
}


////////////////////
//  Dynamic Komi  //
////////////////////
void
DynamicKomi( const game_info_t *game, const uct_node_t *root, const int color )
{
  if (handicap_num != 0) {
    switch(dk_mode) {
      case DK_LINEAR:
	LinearHandicap(game);
	break;
      case DK_VALUE:
	ValueSituational(root, color);
	break;
      default:
	break;
    }
  }
}


////////////////////////////////////////////////////
//    //
////////////////////////////////////////////////////
static void
LinearHandicap( const game_info_t *game )
{
  double new_komi;

  if (game->moves > LINEAR_THRESHOLD - 15) {
  // 
    new_komi = (double)handicap_num + 0.5;
  } else {
    // 
    new_komi = HANDICAP_WEIGHT * handicap_num * (1.0 - ((double)game->moves / LINEAR_THRESHOLD));
  }
  // 
  dynamic_komi[0] = new_komi;
  dynamic_komi[S_BLACK] = new_komi + 1;
  dynamic_komi[S_WHITE] = new_komi - 1;

  PrintKomiValue();
}


//////////////////////////////////
//    //
//////////////////////////////////
static void
ValueSituational( const uct_node_t *root, const int color )
{
  double win_rate = (double)root->win / root->move_count;

  // 
  if (color == S_BLACK) {
    if (win_rate < RED) {
      dynamic_komi[0]--;
    } else if (win_rate > GREEN) {
      dynamic_komi[0]++;
    }
  } else if (color == S_WHITE) {
    if (win_rate < RED) {
      dynamic_komi[0]++;
    } else if (win_rate > GREEN) {
      dynamic_komi[0]--;
    }
  }

  dynamic_komi[S_BLACK] = dynamic_komi[0] + 1.0;
  dynamic_komi[S_WHITE] = dynamic_komi[0] - 1.0;

  PrintKomiValue();
}
