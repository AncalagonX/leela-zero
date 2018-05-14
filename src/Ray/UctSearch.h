#ifndef _UCTSEARCH_H_
#define _UCTSEARCH_H_

#include <atomic>
#include <random>

#include "GoBoard.h"
#include "ZobristHash.h"

////////////
//    //
////////////

const int THREAD_MAX = 32;              // 
const int MAX_NODES = 1000000;          // UCT
const double ALL_THINKING_TIME = 90.0;  // ()
const int CONST_PLAYOUT = 10000;        // 1()
const double CONST_TIME = 10.0;         // 1()
const int PLAYOUT_SPEED = 1000;         // 

// 
const int TIME_RATE_9 = 20;
const int TIME_C_13 = 30;
const int TIME_MAXPLY_13 = 30;
const int TIME_C_19 = 60;
const int TIME_MAXPLY_19 = 80;

// CriticalityOwner
const int CRITICALITY_INTERVAL = 100;

// 
const double FPU = 5.0;

// Progressive Widening
const double PROGRESSIVE_WIDENING = 1.8;

// 
const int EXPAND_THRESHOLD_9  = 20;
const int EXPAND_THRESHOLD_13 = 25;
const int EXPAND_THRESHOLD_19 = 40;


// ( + )
const int UCT_CHILD_MAX = PURE_BOARD_MAX + 1;

// 
const int NOT_EXPANDED = -1;

// 
const int PASS_INDEX = 0;

// UCB Bonus
const double BONUS_EQUIVALENCE = 1000;
const double BONUS_WEIGHT = 0.35;

// 
const double PASS_THRESHOLD = 0.90;
// 
const double RESIGN_THRESHOLD = 0.20;

// Virtual Loss (Best Parameter)
const int VIRTUAL_LOSS = 1;

enum SEARCH_MODE {
  CONST_PLAYOUT_MODE,             // 1
  CONST_TIME_MODE,                // 1
  TIME_SETTING_MODE,              // ()
  TIME_SETTING_WITH_BYOYOMI_MODE, // ()
};


//////////////
//    //
//////////////
struct thread_arg_t {
  game_info_t *game; // 
  int thread_id;   // 
  int color;       // 
};

struct statistic_t {
  std::atomic<int> colors[3];  // 
};

struct child_node_t {
  int pos;  // 
  std::atomic<int> move_count;  // 
  std::atomic<int> win;         // 
  int index;   // 
  double rate; // 
  bool flag;   // Progressive Widening
  bool open;   // 
  bool ladder; // 
};

//  9x9  : 1828bytes
// 13x13 : 3764bytes
// 19x19 : 7988bytes
struct uct_node_t {
  int previous_move1;                 // 1
  int previous_move2;                 // 2
  std::atomic<int> move_count;
  std::atomic<int> win;
  int width;                          // 
  int child_num;                      // 
  child_node_t child[UCT_CHILD_MAX];  // 
  statistic_t statistic[BOARD_MAX];   //  
  bool seki[BOARD_MAX];
};

struct po_info_t {
  int num;   // 
  int halt;  // 
  std::atomic<int> count;       // 
};

struct rate_order_t {
  int index;    // 
  double rate;  // 
};


//////////////////////
//    //
//////////////////////

// 
extern double remaining_time[S_MAX];
// UCT
extern uct_node_t *uct_node;

// 
extern int current_root;

// Criticality
extern double criticality[BOARD_MAX]; 


////////////
//    //
////////////

// 
void StopPondering( void );

// 
void SetPonderingMode( bool flag );

// 
void SetMode( enum SEARCH_MODE mode );

// 1
void SetPlayout( int po );

// 1
void SetConstTime( double time );

// 
void SetThread( int new_thread );

// 
void SetTime( double time );

// 
void SetParameter( void );

// time_settings
void SetTimeSettings( int main_time, int byoyomi, int stones );

// UCT
void InitializeUctSearch( void ); 

// 
void InitializeSearchSetting( void );

// UCT
int UctSearchGenmove( game_info_t *game, int color );

// 
void UctSearchPondering( game_info_t *game, int color );

// UCT
int UctAnalyze( game_info_t *game, int color );

// dest
void OwnerCopy( int *dest );

// Criticaltydest
void CopyCriticality( double *dest );

void CopyStatistic( statistic_t *dest );

// UCT(Clean Up)
int UctSearchGenmoveCleanUp( game_info_t *game, int color );

// 
void SetReuseSubtree( bool flag );


#endif
