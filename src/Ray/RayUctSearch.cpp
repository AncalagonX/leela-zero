#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <random>

#include "DynamicKomi.h"
#include "GoBoard.h"
#include "Ladder.h"
#include "Message.h"
#include "PatternHash.h"
#include "Seki.h"
#include "Simulation.h"
#include "UctRating.h"
#include "UctSearch.h"
#include "Utility.h"

#if defined (_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#endif

using namespace std;

#define LOCK_NODE(var) mutex_nodes[(var)].lock()
#define UNLOCK_NODE(var) mutex_nodes[(var)].unlock()
#define LOCK_EXPAND mutex_expand.lock();
#define UNLOCK_EXPAND mutex_expand.unlock();

////////////////
//    //
////////////////

// 
double remaining_time[S_MAX];

// UCT
uct_node_t *uct_node;

// 
static po_info_t po_info;

// Progressive Widening 
static int pw[PURE_BOARD_MAX + 1];  

// 
static int expand_threshold = EXPAND_THRESHOLD_19;

// 
static bool extend_time = false;

int current_root; // 
mutex mutex_nodes[MAX_NODES];
mutex mutex_expand;       // mutex

// 
enum SEARCH_MODE mode = CONST_TIME_MODE;
// 
int threads = 1;
// 1
double const_thinking_time = CONST_TIME;
// 1
int playout = CONST_PLAYOUT;
// 
double default_remaining_time = ALL_THINKING_TIME;

// 
thread_arg_t t_arg[THREAD_MAX];

// 
statistic_t statistic[BOARD_MAX];  
// Criticality
double criticality[BOARD_MAX];  
// Owner(0-100%)
double owner[BOARD_MAX];  

// 
int owner_index[BOARD_MAX];   
// 
int criticality_index[BOARD_MAX];  

// 
bool candidates[BOARD_MAX];  

bool pondering_mode = false;

bool ponder = false;

bool pondering_stop = false;

bool pondered = false;

double time_limit;

std::thread *handle[THREAD_MAX];    // 

// UCB Bonus
double bonus_equivalence = BONUS_EQUIVALENCE;
// UCB Bonus
double bonus_weight = BONUS_WEIGHT;

// 
std::mt19937_64 *mt[THREAD_MAX];

// Criticality
int criticality_max = CRITICALITY_MAX;

// 
bool reuse_subtree = false;

// 
int my_color;

ray_clock::time_point begin_time;


////////////
//    //
////////////

// Virtual Loss
static void AddVirtualLoss( child_node_t *child, int current );

// 
static void CalculateNextPlayouts( game_info_t *game, int color, double best_wp, double finish_time );

// Criticaliity
static void CalculateCriticality( int color );

// Criticality
static void CalculateCriticalityIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index );

// Ownership
static void CalculateOwner( int color, int count );

// Ownership
static void CalculateOwnerIndex( uct_node_t *node, statistic_t *node_statistc, int color, int *index );

// 
static void CorrectDescendentNodes( vector<int> &indexes, int index );

// 
static int ExpandNode( game_info_t *game, int color, int current );

// 
static int ExpandRoot( game_info_t *game, int color );

// 
static bool ExtendTime( void );

// 
static void InitializeCandidate( child_node_t *uct_child, int pos, bool ladder );

// 
static bool InterruptionCheck( void );

// UCT
static void ParallelUctSearch( thread_arg_t *arg );

// UCT()
static void ParallelUctSearchPondering( thread_arg_t *arg );

// 
static void RatingNode( game_info_t *game, int color, int index );

static int RateComp( const void *a, const void *b );

// UCB
static int SelectMaxUcbChild( int current, int color );

// 
static void Statistic( game_info_t *game, int winner );

// UCT(1, 1)
static int UctSearch( game_info_t *game, int color, std::mt19937_64 *mt, int current, int *winner );

// 
static void UpdateNodeStatistic( game_info_t *game, int winner, statistic_t *node_statistic );

// 
static void UpdateResult( child_node_t *child, int result, int current );



/////////////////////
//    //
/////////////////////
void
SetPonderingMode( bool flag )
{
  pondering_mode = flag;
}


////////////////////////
//    //
////////////////////////
void
SetMode( enum SEARCH_MODE new_mode )
{
  mode = new_mode;
}


///////////////////////////////////////
//  1  //
///////////////////////////////////////
void
SetPlayout( int po )
{
  playout = po;
}


/////////////////////////////////
//  1  //
/////////////////////////////////
void
SetConstTime( double time )
{
  const_thinking_time = time;
}


////////////////////////////////
//    //
////////////////////////////////
void
SetThread( int new_thread )
{
  threads = new_thread;
}


//////////////////////
//    //
//////////////////////
void
SetTime( double time )
{
  default_remaining_time = time;
}


//////////////////////////
//    //
//////////////////////////
void
SetReuseSubtree( bool flag )
{
  reuse_subtree = flag;
}


////////////////////////////////////////////
//    //
////////////////////////////////////////////
void
SetParameter( void )
{
  if (pure_board_size < 11) {
    expand_threshold = EXPAND_THRESHOLD_9;
  } else if (pure_board_size < 16) {
    expand_threshold = EXPAND_THRESHOLD_13;
  } else {
    expand_threshold = EXPAND_THRESHOLD_19;
  }
}

//////////////////////////////////////
//  time_settings  //
//////////////////////////////////////
void
SetTimeSettings( int main_time, int byoyomi, int stone )
{
  if (mode == CONST_PLAYOUT_MODE ||
      mode == CONST_TIME_MODE) {
    return ;
  }
  
  if (main_time == 0) {
    const_thinking_time = (double)byoyomi * 0.85;
    mode = CONST_TIME_MODE;
    cerr << "Const Thinking Time Mode" << endl;
  } else {
    if (byoyomi == 0) {
      default_remaining_time = main_time;
      mode = TIME_SETTING_MODE;
      cerr << "Time Setting Mode" << endl;
    } else {
      default_remaining_time = main_time;
      const_thinking_time = ((double)byoyomi) / stone;
      mode = TIME_SETTING_WITH_BYOYOMI_MODE;
      cerr << "Time Setting Mode (byoyomi)" << endl;
    }
  }
}

/////////////////////////
//  UCT  //
/////////////////////////
void
InitializeUctSearch( void )
{
  int i;

  // Progressive Widening  
  pw[0] = 0;
  for (i = 1; i <= PURE_BOARD_MAX; i++) {  
    pw[i] = pw[i - 1] + (int)(40 * pow(PROGRESSIVE_WIDENING, i - 1));
    if (pw[i] > 10000000) break;
  }
  for (i = i + 1; i <= PURE_BOARD_MAX; i++) { 
    pw[i] = INT_MAX;
  }

  // UCT
  uct_node = new uct_node_t[uct_hash_size];
  
  if (uct_node == NULL) {
    cerr << "Cannot allocate memory !!" << endl;
    cerr << "You must reduce tree size !!" << endl;
    exit(1);
  }

}


////////////////////////
//    //
////////////////////////
void
InitializeSearchSetting( void )
{
  // Owner
  for (int i = 0; i < board_max; i++){
    owner[i] = 50;
    owner_index[i] = 5;
    candidates[i] = true;
  }

  // 
  for (int i = 0; i < THREAD_MAX; i++) {
    if (mt[i]) {
      delete mt[i];
    }
    mt[i] = new mt19937_64((unsigned int)(time(NULL) + i));
  }

  // 
  for (int i = 0; i < 3; i++) {
    remaining_time[i] = default_remaining_time;
  }

  // 
  // 
  if (mode == CONST_PLAYOUT_MODE) {
    time_limit = 100000.0;
    po_info.num = playout;
    extend_time = false;
  } else if (mode == CONST_TIME_MODE) {
    time_limit = const_thinking_time;
    po_info.num = 100000000;
    extend_time = false;
  } else if (mode == TIME_SETTING_MODE ||
	     mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
    if (pure_board_size < 11) {
      time_limit = remaining_time[0] / TIME_RATE_9;
      po_info.num = (int)(PLAYOUT_SPEED * time_limit);
      extend_time = true;
    } else if (pure_board_size < 13) {
      time_limit = remaining_time[0] / (TIME_MAXPLY_13 + TIME_C_13);
      po_info.num = (int)(PLAYOUT_SPEED * time_limit);
      extend_time = true;
    } else {
      time_limit = remaining_time[0] / (TIME_MAXPLY_19 + TIME_C_19);
      po_info.num = (int)(PLAYOUT_SPEED * time_limit);
      extend_time = true;
    }
  }

  pondered = false;
  pondering_stop = true;
}


void
StopPondering( void )
{
  if (!pondering_mode) {
    return ;
  }

  if (ponder) {
    pondering_stop = true;
    for (int i = 0; i < threads; i++) {
      handle[i]->join();
      delete handle[i];
    }
    ponder = false;
    pondered = true;
    PrintPonderingCount(po_info.count);
  }
}


/////////////////////////////////////
//  UCT  //
/////////////////////////////////////
int
UctSearchGenmove( game_info_t *game, int color )
{
  int pos, select_index, max_count, pre_simulated;
  double finish_time, pass_wp, best_wp;
  child_node_t *uct_child;

  // 
  if (!pondered) {
    memset(statistic, 0, sizeof(statistic_t) * board_max);
    fill_n(criticality_index, board_max, 0);
    for (int i = 0; i < board_max; i++) {
      criticality[i] = 0.0;
    }
  }
  po_info.count = 0;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    owner[pos] = 50;
    owner_index[pos] = 5;
    candidates[pos] = true;
  }

  if (!reuse_subtree) {
    ClearUctHash();
  }
  
  // 
  begin_time = ray_clock::now();
  
  // UCT
  current_root = ExpandRoot(game, color);

  // 
  pre_simulated = uct_node[current_root].move_count;

  // 1()PASS
  if (uct_node[current_root].child_num <= 1) {
    return PASS;
  }

  // 
  po_info.halt = po_info.num;

  // 
  my_color = color;

  // Dynamic Komi()
  DynamicKomi(game, &uct_node[current_root], color);

  // 
  PrintPlayoutLimits(time_limit, po_info.halt);

  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle[i] = new thread(ParallelUctSearch, &t_arg[i]);
  }

  for (int i = 0; i < threads; i++) {
    handle[i]->join();
    delete handle[i];
  }
  // 41, 
  // ,
  // 
  // 1.5
  if (game->moves > pure_board_size * 3 - 17 &&
      extend_time &&
      ExtendTime()) {
    po_info.halt = (int)(1.5 * po_info.halt);
    time_limit *= 1.5;
    for (int i = 0; i < threads; i++) {
      handle[i] = new thread(ParallelUctSearch, &t_arg[i]);
    }

    for (int i = 0; i < threads; i++) {
      handle[i]->join();
      delete handle[i];
    }
  }

  uct_child = uct_node[current_root].child;

  select_index = PASS_INDEX;
  max_count = uct_child[PASS_INDEX].move_count;

  // 
  for (int i = 1; i < uct_node[current_root].child_num; i++){
    if (uct_child[i].move_count > max_count) {
      select_index = i;
      max_count = uct_child[i].move_count;
    }
  }

  // 
  finish_time = GetSpendTime(begin_time);

  // 
  if (uct_child[PASS_INDEX].move_count != 0) {
    pass_wp = (double)uct_child[PASS_INDEX].win / uct_child[PASS_INDEX].move_count;
  } else {
    pass_wp = 0;
  }

  // (Dynamic Komi)
  best_wp = (double)uct_child[select_index].win / uct_child[select_index].move_count;

  // 
  PrintOwner(&uct_node[current_root], color, owner);

  // 
  // 1. , PASS_THRESHOLD
  // 2. MAX_MOVES
  // 
  //    Dynamic KomiRESIGN_THRESHOLD
  // 
  if (pass_wp >= PASS_THRESHOLD &&
      (game->record[game->moves - 1].pos == PASS)){
    pos = PASS;
  } else if (game->moves >= MAX_MOVES) {
    pos = PASS;
  } else if (game->moves > 3 &&
	     game->record[game->moves - 1].pos == PASS &&
	     game->record[game->moves - 3].pos == PASS) {
    pos = PASS;
  } else if (best_wp <= RESIGN_THRESHOLD) {
    pos = RESIGN;
  } else {
    pos = uct_child[select_index].pos;
  }

  // 
  PrintBestSequence(game, uct_node, current_root, color);
  // (, , , , )
  PrintPlayoutInformation(&uct_node[current_root], &po_info, finish_time, pre_simulated);
  // 
  CalculateNextPlayouts(game, color, best_wp, finish_time);

  return pos;
}


///////////////
//    //
///////////////
void
UctSearchPondering( game_info_t *game, int color )
{
  int pos;

  if (!pondering_mode) {
    return ;
  }

  // 
  memset(statistic, 0, sizeof(statistic_t) * board_max);  
  fill_n(criticality_index, board_max, 0);  
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;    
  }
				  
  po_info.count = 0;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    owner[pos] = 50;
    owner_index[pos] = 5;
    candidates[pos] = true;
  }

  // UCT
  current_root = ExpandRoot(game, color);

  pondered = false;

  // 1()PASS
  if (uct_node[current_root].child_num <= 1) {
    ponder = false;
    pondering_stop = true;
    return ;
  }

  ponder = true;
  pondering_stop = false;

  // Dynamic Komi()
  DynamicKomi(game, &uct_node[current_root], color);

  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle[i] = new thread(ParallelUctSearchPondering, &t_arg[i]);
  }

  return ;
}

/////////////////////
//    //
/////////////////////
static void
InitializeCandidate( child_node_t *uct_child, int pos, bool ladder )
{
  uct_child->pos = pos;
  uct_child->move_count = 0;
  uct_child->win = 0;
  uct_child->index = NOT_EXPANDED;
  uct_child->rate = 0.0;
  uct_child->flag = false;
  uct_child->open = false;
  uct_child->ladder = ladder;
}


/////////////////////////
//    //
/////////////////////////
static int
ExpandRoot( game_info_t *game, int color )
{
  unsigned long long hash = game->move_hash;
  unsigned int index = FindSameHashIndex(hash, color, game->moves);
  child_node_t *uct_child;
  int i, pos, child_num = 0;
  bool ladder[BOARD_MAX] = { false };  
  int pm1 = PASS, pm2 = PASS;
  int moves = game->moves;

  // 
  pm1 = game->record[moves - 1].pos;
  // 2
  if (moves > 1) pm2 = game->record[moves - 2].pos;

  // 9  
  if (pure_board_size != 9) {
    LadderExtension(game, color, ladder);
  }

  // , 
  if (index != uct_hash_size) {
    vector<int> indexes;

    // 
    CorrectDescendentNodes(indexes, index);
    std::sort(indexes.begin(), indexes.end());
    ClearNotDescendentNodes(indexes);
    
    // 2
    uct_node[index].previous_move1 = pm1;
    uct_node[index].previous_move2 = pm2;

    uct_child = uct_node[index].child;

    child_num = uct_node[index].child_num;

    for (i = 0; i < child_num; i++) {
      pos = uct_child[i].pos;
      uct_child[i].rate = 0.0;
      uct_child[i].flag = false;
      uct_child[i].open = false;
      if (ladder[pos]) {
	uct_node[index].move_count -= uct_child[i].move_count;
	uct_node[index].win -= uct_child[i].win;
	uct_child[i].move_count = 0;
	uct_child[i].win = 0;
      }
      uct_child[i].ladder = ladder[pos];
    }

    // 1
    uct_node[index].width = 1;

    // 
    RatingNode(game, color, index);

    PrintReuseCount(uct_node[index].move_count);

    return index;
  } else {
    // 
    ClearUctHash();
    
    // 
    index = SearchEmptyIndex(hash, color, game->moves);

    assert(index != uct_hash_size);    
    
    // 
    uct_node[index].previous_move1 = pm1;
    uct_node[index].previous_move2 = pm2;
    uct_node[index].move_count = 0;
    uct_node[index].win = 0;
    uct_node[index].width = 0;
    uct_node[index].child_num = 0;
    memset(uct_node[index].statistic, 0, sizeof(statistic_t) * BOARD_MAX); 
    fill_n(uct_node[index].seki, BOARD_MAX, false);
    
    uct_child = uct_node[index].child;
    
    // 
    InitializeCandidate(&uct_child[PASS_INDEX], PASS, ladder[PASS]);
    child_num++;
    
    // 
    if (game->moves == 1) {
      for (i = 0; i < first_move_candidates; i++) {
	pos = first_move_candidate[i];
	// 
	if (candidates[pos] && IsLegal(game, pos, color)) {
	  InitializeCandidate(&uct_child[child_num], pos, ladder[pos]);
	  child_num++;
	}	
      }
    } else {
      for (i = 0; i < pure_board_max; i++) {
	pos = onboard_pos[i];
	// 
	if (candidates[pos] && IsLegal(game, pos, color)) {
	  InitializeCandidate(&uct_child[child_num], pos, ladder[pos]);
	  child_num++;
	}
      }
    }
    
    // 
    uct_node[index].child_num = child_num;
    
    // 
    RatingNode(game, color, index);

    // 
    CheckSeki(game, uct_node[index].seki);
    
    uct_node[index].width++;
  }

  return index;
}



///////////////////
//    //
///////////////////
static int
ExpandNode( game_info_t *game, int color, int current )
{
  unsigned long long hash = game->move_hash;
  unsigned int index = FindSameHashIndex(hash, color, game->moves);
  child_node_t *uct_child, *uct_sibling;
  int i, pos, child_num = 0;
  double max_rate = 0.0;
  int max_pos = PASS, sibling_num;
  int pm1 = PASS, pm2 = PASS;
  int moves = game->moves;

  // , 
  if (index != uct_hash_size) {
    return index;
  }

  // 
  index = SearchEmptyIndex(hash, color, game->moves);

  assert(index != uct_hash_size);    

  // 
  pm1 = game->record[moves - 1].pos;
  // 2
  if (moves > 1) pm2 = game->record[moves - 2].pos;

  // 
  uct_node[index].previous_move1 = pm1;
  uct_node[index].previous_move2 = pm2;
  uct_node[index].move_count = 0;
  uct_node[index].win = 0;
  uct_node[index].width = 0;
  uct_node[index].child_num = 0;
  memset(uct_node[index].statistic, 0, sizeof(statistic_t) * BOARD_MAX);  
  fill_n(uct_node[index].seki, BOARD_MAX, false);
  uct_child = uct_node[index].child;

  // 
  InitializeCandidate(&uct_child[PASS_INDEX], PASS, false);
  child_num++;

  // 
  for (i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    // 
    if (candidates[pos] && IsLegal(game, pos, color)) {
      InitializeCandidate(&uct_child[child_num], pos, false);
      child_num++;
    }
  }

  // 
  uct_node[index].child_num = child_num;

  // 
  RatingNode(game, color, index);

  // 
  CheckSeki(game, uct_node[index].seki);
  
  // 1
  uct_node[index].width++;

  // 
  uct_sibling = uct_node[current].child;
  sibling_num = uct_node[current].child_num;
  for (i = 0; i < sibling_num; i++) {
    if (uct_sibling[i].pos != pm1) {
      if (uct_sibling[i].rate > max_rate) {
	max_rate = uct_sibling[i].rate;
	max_pos = uct_sibling[i].pos;
      }
    }
  }

  // 
  for (i = 0; i < child_num; i++) {
    if (uct_child[i].pos == max_pos) {
      if (!uct_child[i].flag) {
	uct_child[i].open = true;
      }
      break;
    }
  }

  return index;
}


//////////////////////////////////////
//               //
//  (Progressive Widening)  //
//////////////////////////////////////
static void
RatingNode( game_info_t *game, int color, int index )
{
  int child_num = uct_node[index].child_num;
  int pos;
  int moves = game->moves;
  double score = 0.0;
  int max_index;
  double max_score;
  pattern_hash_t hash_pat;
  int pat_index[3] = {0};
  double dynamic_parameter;
  bool self_atari_flag;
  child_node_t *uct_child = uct_node[index].child;
  uct_features_t uct_features;

  memset(&uct_features, 0, sizeof(uct_features_t));

  // 
  uct_child[PASS_INDEX].rate = CalculateLFRScore(game, PASS, pat_index, &uct_features);

  // 
  UctCheckFeatures(game, color, &uct_features);
  // 2
  UctCheckRemove2Stones(game, color, &uct_features);
  // 3
  UctCheckRemove3Stones(game, color, &uct_features);
  // 2, 
  if (game->ko_move == moves - 2) {
    UctCheckCaptureAfterKo(game, color, &uct_features);
    UctCheckKoConnection(game, &uct_features);
  }

  max_index = 0;
  max_score = uct_child[0].rate;

  for (int i = 1; i < child_num; i++) {
    pos = uct_child[i].pos;

    // 
    self_atari_flag = UctCheckSelfAtari(game, color, pos, &uct_features);
    // 
    UctCheckSnapBack(game, color, pos, &uct_features);
    // 
    if ((uct_features.tactical_features1[pos] & capture_mask)== 0) {
      UctCheckCapture(game, color, pos, &uct_features);
    }
    // 
    if ((uct_features.tactical_features1[pos] & atari_mask) == 0) {
      UctCheckAtari(game, color, pos, &uct_features);
    }
    // 
    UctCheckDoubleKeima(game, color, pos, &uct_features);
    // 
    UctCheckKeimaTsukekoshi(game, color, pos, &uct_features);

    // 0.0
    // -1.0
    if (!self_atari_flag) {
      score = 0.0;
    } else if (uct_child[i].ladder) {
      score = -1.0;
    } else {
      // MD3, MD4, MD5
      PatternHash(&game->pat[pos], &hash_pat);
      // MD3
      pat_index[0] = SearchIndex(md3_index, hash_pat.list[MD_3]);
      // MD4
      pat_index[1] = SearchIndex(md4_index, hash_pat.list[MD_4]);
      // MD5
      pat_index[2] = SearchIndex(md5_index, hash_pat.list[MD_5 + MD_MAX]);

      score = CalculateLFRScore(game, pos, pat_index, &uct_features);
    }

    // 
    uct_child[i].rate = score;

    // OwnerCriticality
    dynamic_parameter = uct_owner[owner_index[pos]] + uct_criticality[criticality_index[pos]];

    // 
    if (score + dynamic_parameter > max_score) {
      max_index = i;
      max_score = score + dynamic_parameter;
    }
  }

  // 
  uct_child[max_index].flag = true;
}




//////////////////////////
//    //
//////////////////////////
static bool
InterruptionCheck( void )
{
  int max = 0, second = 0;
  const int child_num = uct_node[current_root].child_num;
  const int rest = po_info.halt - po_info.count;
  child_node_t *uct_child = uct_node[current_root].child;

  if (mode != CONST_PLAYOUT_MODE && 
      GetSpendTime(begin_time) * 10.0 < time_limit) {
      return false;
  }

  // 
  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].move_count > max) {
      second = max;
      max = uct_child[i].move_count;
    } else if (uct_child[i].move_count > second) {
      second = uct_child[i].move_count;
    }
  }

  // 
  // 
  if (max - second > rest) {
    return true;
  } else {
    return false;
  }
}


///////////////////////////
//     //
///////////////////////////
static bool
ExtendTime( void )
{
  int max = 0, second = 0;
  const int child_num = uct_node[current_root].child_num;
  child_node_t *uct_child = uct_node[current_root].child;

  // 
  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].move_count > max) {
      second = max;
      max = uct_child[i].move_count;
    } else if (uct_child[i].move_count > second) {
      second = uct_child[i].move_count;
    }
  }

  // 
  // 1.2
  if (max < second * 1.2) {
    return true;
  } else {
    return false;
  }
}



/////////////////////////////////
//       //
//  UCT  //
/////////////////////////////////
static void
ParallelUctSearch( thread_arg_t *arg )
{
  thread_arg_t *targ = (thread_arg_t *)arg;
  game_info_t *game;
  int color = targ->color;
  bool interruption = false;
  bool enough_size = true;
  int winner = 0;
  int interval = CRITICALITY_INTERVAL;
  
  game = AllocateGame();

  // ID0
  // , 
  if (targ->thread_id == 0) {
    do {
      // 1	
      atomic_fetch_add(&po_info.count, 1);
      // 
      CopyGame(game, targ->game);
      // 1
      UctSearch(game, color, mt[targ->thread_id], current_root, &winner);
      // 
      interruption = InterruptionCheck();
      // 
      enough_size = CheckRemainingHashSize();
      // OwnerCriticality
      if (po_info.count > interval) {
	CalculateOwner(color, po_info.count);
	CalculateCriticality(color);
	interval += CRITICALITY_INTERVAL;
      }
      if (GetSpendTime(begin_time) > time_limit) break;
    } while (po_info.count < po_info.halt && !interruption && enough_size);
  } else {
    do {
      // 1	
      atomic_fetch_add(&po_info.count, 1);
      // 
      CopyGame(game, targ->game);
      // 1
      UctSearch(game, color, mt[targ->thread_id], current_root, &winner);
      // 
      interruption = InterruptionCheck();
      // 
      enough_size = CheckRemainingHashSize();
      if (GetSpendTime(begin_time) > time_limit) break;
    } while (po_info.count < po_info.halt && !interruption && enough_size);
  }

  // 
  FreeGame(game);
  return;
}


/////////////////////////////////
//       //
//  UCT  //
/////////////////////////////////
static void
ParallelUctSearchPondering( thread_arg_t *arg )
{
  thread_arg_t *targ = (thread_arg_t *)arg;
  game_info_t *game;
  int color = targ->color;
  bool enough_size = true;
  int winner = 0;
  int interval = CRITICALITY_INTERVAL;

  game = AllocateGame();

  // ID0
  // , 
  if (targ->thread_id == 0) {
    do {
      // 1	
      atomic_fetch_add(&po_info.count, 1);
      // 
      CopyGame(game, targ->game);
      // 1
      UctSearch(game, color, mt[targ->thread_id], current_root, &winner);
      // 
      enough_size = CheckRemainingHashSize();
      // OwnerCriticality
      if (po_info.count > interval) {
	CalculateOwner(color, po_info.count);
	CalculateCriticality(color);
	interval += CRITICALITY_INTERVAL;
      }
    } while (!pondering_stop && enough_size);
  } else {
    do {
      // 1	
      atomic_fetch_add(&po_info.count, 1);
      // 
      CopyGame(game, targ->game);
      // 1
      UctSearch(game, color, mt[targ->thread_id], current_root, &winner);
      // 
      enough_size = CheckRemainingHashSize();
    } while (!pondering_stop && enough_size);
  }

  // 
  FreeGame(game);
  return;
}


//////////////////////////////////////////////
//  UCT                        //
//  1, 1    //
//////////////////////////////////////////////
static int 
UctSearch( game_info_t *game, int color, mt19937_64 *mt, int current, int *winner )
{
  int result = 0, next_index;
  double score;
  child_node_t *uct_child = uct_node[current].child;  

  // 
  LOCK_NODE(current);
  // UCB
  next_index = SelectMaxUcbChild(current, color);
  // 
  PutStone(game, uct_child[next_index].pos, color);
  // 
  color = FLIP_COLOR(color);

  if (uct_child[next_index].move_count < expand_threshold) {
    // Virtual Loss
    AddVirtualLoss(&uct_child[next_index], current);

    memcpy(game->seki, uct_node[current].seki, sizeof(bool) * BOARD_MAX);
    
    // 
    UNLOCK_NODE(current);

    // 
    Simulation(game, color, mt);
    
    // 
    score = (double)CalculateScore(game);
    
    // 
    if (my_color == S_BLACK) {
      if (score - dynamic_komi[my_color] >= 0) {
	result = (color == S_BLACK ? 0 : 1);
	*winner = S_BLACK;
      } else {
	result = (color == S_WHITE ? 0 : 1);
	*winner = S_WHITE;
      }
    } else {
      if (score - dynamic_komi[my_color] > 0) {
	result = (color == S_BLACK ? 0 : 1);
	*winner = S_BLACK;
      } else {
	result = (color == S_WHITE ? 0 : 1);
	*winner = S_WHITE;
      }
    }
    // 
    Statistic(game, *winner);
  } else {
    // Virtual Loss
    AddVirtualLoss(&uct_child[next_index], current);
    // 
    if (uct_child[next_index].index == -1) {
      // 
      LOCK_EXPAND;
      // 
      uct_child[next_index].index = ExpandNode(game, color, current);
      // 
      UNLOCK_EXPAND;
    }
    // 
    UNLOCK_NODE(current);
    // 1
    result = UctSearch(game, color, mt, uct_child[next_index].index, winner);
  }

  // 
  UpdateResult(&uct_child[next_index], result, current);

  // 
  UpdateNodeStatistic(game, *winner, uct_node[current].statistic);

  return 1 - result;
}


//////////////////////////
//  Virtual Loss  //
//////////////////////////
static void
AddVirtualLoss(child_node_t *child, int current)
{
#if defined CPP11
  atomic_fetch_add(&uct_node[current].move_count, VIRTUAL_LOSS);
  atomic_fetch_add(&child->move_count, VIRTUAL_LOSS);
#else
  uct_node[current].move_count += VIRTUAL_LOSS;
  child->move_count += VIRTUAL_LOSS;
#endif
}


//////////////////////
//    //
/////////////////////
static void
UpdateResult( child_node_t *child, int result, int current )
{
  atomic_fetch_add(&uct_node[current].win, result);
  atomic_fetch_add(&uct_node[current].move_count, 1 - VIRTUAL_LOSS);
  atomic_fetch_add(&child->win, result);
  atomic_fetch_add(&child->move_count, 1 - VIRTUAL_LOSS);
}


//////////////////////////
//    //
//////////////////////////
static int
RateComp( const void *a, const void *b )
{
  rate_order_t *ro1 = (rate_order_t *)a;
  rate_order_t *ro2 = (rate_order_t *)b;
  if (ro1->rate < ro2->rate) {
    return 1;
  } else if (ro1->rate > ro2->rate) {
    return -1;
  } else {
    return 0;
  }
}


/////////////////////////////////////////////////////
//  UCB  //
/////////////////////////////////////////////////////
static int
SelectMaxUcbChild( int current, int color )
{
  child_node_t *uct_child = uct_node[current].child;
  const int child_num = uct_node[current].child_num;
  int max_child = 0;
  const int sum = uct_node[current].move_count;
  double p, max_value;
  double ucb_value;
  int max_index;
  double max_rate;
  double dynamic_parameter;
  rate_order_t order[PURE_BOARD_MAX + 1];  
  int pos;
  int width;
  const double ucb_bonus_weight = bonus_weight * sqrt(bonus_equivalence / (sum + bonus_equivalence));

  // 128OwnerCriticality  
  if ((sum & 0x7f) == 0 && sum != 0) {
    int o_index[UCT_CHILD_MAX], c_index[UCT_CHILD_MAX];
    CalculateCriticalityIndex(&uct_node[current], uct_node[current].statistic, color, c_index);
    CalculateOwnerIndex(&uct_node[current], uct_node[current].statistic, color, o_index);
    for (int i = 0; i < child_num; i++) {
      pos = uct_child[i].pos;
      if (pos == PASS) {
	dynamic_parameter = 0.0;
      } else {
	dynamic_parameter = uct_owner[o_index[i]] + uct_criticality[c_index[i]];
      }
      order[i].rate = uct_child[i].rate + dynamic_parameter;
      order[i].index = i;
      uct_child[i].flag = false;
    }
    qsort(order, child_num, sizeof(rate_order_t), RateComp);

    // 
    width = ((uct_node[current].width > child_num) ? child_num : uct_node[current].width);

    // 
    for (int i = 0; i < width; i++) {
      uct_child[order[i].index].flag = true;
    }
  }
  	
  // Progressive Widening, 
  // 1
  if (sum > pw[uct_node[current].width]) {
    max_index = -1;
    max_rate = 0;
    for (int i = 0; i < child_num; i++) {
      if (uct_child[i].flag == false) {
	pos = uct_child[i].pos;
	dynamic_parameter = uct_owner[owner_index[pos]] + uct_criticality[criticality_index[pos]];
	if (uct_child[i].rate + dynamic_parameter > max_rate) {
	  max_index = i;
	  max_rate = uct_child[i].rate + dynamic_parameter;
	}
      }
    }
    if (max_index != -1) {
      uct_child[max_index].flag = true;
    }
    uct_node[current].width++;  
  }

  max_value = -1;
  max_child = 0;

  // UCB  
  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].flag || uct_child[i].open) {
      if (uct_child[i].move_count == 0) {
	ucb_value = FPU;
      } else {
	double div, v;
	// UCB1-TUNED value
	p = (double)uct_child[i].win / uct_child[i].move_count;
	div = log(sum) / uct_child[i].move_count;
	v = p - p * p + sqrt(2.0 * div);
	ucb_value = p + sqrt(div * ((0.25 < v) ? 0.25 : v));

	// UCB Bonus
	ucb_value += ucb_bonus_weight * uct_child[i].rate;
      }

      if (ucb_value > max_value) {
	max_value = ucb_value;
	max_child = i;
      }
    }
  }

  return max_child;
}


///////////////////////////////////////////////////////////
//  OwnerCriiticality  //
///////////////////////////////////////////////////////////
static void
Statistic( game_info_t *game, int winner )
{
  const char *board = game->board;
  int pos, color;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    color = board[pos];
    if (color == S_EMPTY) color = territory[Pat3(game->pat, pos)];

    std::atomic_fetch_add(&statistic[pos].colors[color], 1);
    if (color == winner) {
      std::atomic_fetch_add(&statistic[pos].colors[0], 1);
    }
  }
}


///////////////////////////////
//    //
///////////////////////////////
static void
UpdateNodeStatistic( game_info_t *game, int winner, statistic_t *node_statistic )
{
  const char *board = game->board;
  int pos, color;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    color = board[pos];
    if (color == S_EMPTY) color = territory[Pat3(game->pat, pos)];
    std::atomic_fetch_add(&node_statistic[pos].colors[color], 1);
    if (color == winner) {
      std::atomic_fetch_add(&node_statistic[pos].colors[0], 1);
    }
  }
}


//////////////////////////////////
//  Criticality  //
//////////////////////////////////
static void
CalculateCriticalityIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index )
{
  double win, lose;
  const int other = FLIP_COLOR(color);
  const int count = node->move_count;
  const int child_num = node->child_num;
  int pos;
  double tmp;

  win = (double)node->win / node->move_count;
  lose = 1.0 - win;

  index[0] = 0;

  for (int i = 1; i < child_num; i++) {
    pos = node->child[i].pos;

    tmp = ((double)node_statistic[pos].colors[0] / count) -
      ((((double)node_statistic[pos].colors[color] / count) * win)
       + (((double)node_statistic[pos].colors[other] / count) * lose));
    if (tmp < 0) tmp = 0;
    index[i] = (int)(tmp * 40);
    if (index[i] > criticality_max - 1) index[i] = criticality_max - 1;
  }
}

////////////////////////////////////
//  Criticality   // 
////////////////////////////////////
static void
CalculateCriticality( int color )
{
  int pos;
  double tmp;
  const int other = FLIP_COLOR(color);
  double win, lose;

  win = (double)uct_node[current_root].win / uct_node[current_root].move_count;
  lose = 1.0 - win;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];

    tmp = ((float)statistic[pos].colors[0] / po_info.count) -
      ((((float)statistic[pos].colors[color] / po_info.count)*win)
       + (((float)statistic[pos].colors[other] / po_info.count)*lose));
    criticality[pos] = tmp;
    if (tmp < 0) tmp = 0;
    criticality_index[pos] = (int)(tmp * 40);
    if (criticality_index[pos] > criticality_max - 1) criticality_index[pos] = criticality_max - 1;
  }
}


//////////////////////////////
//  Owner   //
//////////////////////////////
static void
CalculateOwnerIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index )
{
  int pos;
  const int count = node->move_count;
  const int child_num = node->child_num;

  index[0] = 0;

  for (int i = 1; i < child_num; i++){
    pos = node->child[i].pos;
    index[i] = (int)((double)node_statistic[pos].colors[color] * 10.0 / count + 0.5);
    if (index[i] > OWNER_MAX - 1) index[i] = OWNER_MAX - 1;
    if (index[i] < 0)   index[pos] = 0;
  }
}


//////////////////////////////
//  Owner   //
//////////////////////////////
static void
CalculateOwner( int color, int count )
{
  int pos;

  for (int i = 0; i < pure_board_max; i++){
    pos = onboard_pos[i];
    owner_index[pos] = (int)((double)statistic[pos].colors[color] * 10.0 / count + 0.5);
    if (owner_index[pos] > OWNER_MAX - 1) owner_index[pos] = OWNER_MAX - 1;
    if (owner_index[pos] < 0)   owner_index[pos] = 0;
  }
}


/////////////////////////////////
//    //
/////////////////////////////////
static void
CalculateNextPlayouts( game_info_t *game, int color, double best_wp, double finish_time )
{
  double po_per_sec;

  if (finish_time != 0.0) {
    po_per_sec = po_info.count / finish_time;
  } else {
    po_per_sec = PLAYOUT_SPEED * threads;
  }

  // 
  if (mode == CONST_TIME_MODE) {
    if (best_wp > 0.90) {
      po_info.num = (int)(po_info.count / finish_time * const_thinking_time / 2);
    } else {
      po_info.num = (int)(po_info.count / finish_time * const_thinking_time);
    }
  } else if (mode == TIME_SETTING_MODE ||
	     mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
    remaining_time[color] -= finish_time;
    if (pure_board_size < 11) {
      time_limit = remaining_time[color] / TIME_RATE_9;
    } else if (pure_board_size < 16) {
      time_limit = remaining_time[color] / (TIME_C_13 + ((TIME_MAXPLY_13 - (game->moves + 1) > 0) ? TIME_MAXPLY_13 - (game->moves + 1) : 0));
    } else {
      time_limit = remaining_time[color] / (TIME_C_19 + ((TIME_MAXPLY_19 - (game->moves + 1) > 0) ? TIME_MAXPLY_19 - (game->moves + 1) : 0));
    }
    if (mode == TIME_SETTING_WITH_BYOYOMI_MODE &&
	time_limit < (const_thinking_time * 0.5)) {
      time_limit = const_thinking_time * 0.5;
    }
    po_info.num = (int)(po_per_sec * time_limit);	
  } 
}


/////////////////////////////////////
//  UCT  //
/////////////////////////////////////
int
UctAnalyze( game_info_t *game, int color )
{
  int pos;
  thread *handle[THREAD_MAX];

  // 
  memset(statistic, 0, sizeof(statistic_t) * board_max);  
  fill_n(criticality_index, board_max, 0);  
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;
  }
  po_info.count = 0;

  ClearUctHash();

  current_root = ExpandRoot(game, color);

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
  }

  po_info.halt = 10000;

  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle[i] = new std::thread(ParallelUctSearch, &t_arg[i]);
  }


  for (int i = 0; i < threads; i++) {
    handle[i]->join();
    delete handle[i];
  }

  int x, y, black = 0, white = 0;
  double own;

  for (y = board_start; y <= board_end; y++) {
    for (x = board_start; x <= board_end; x++) {
      pos = POS(x, y);
      own = (double)statistic[pos].colors[S_BLACK] / uct_node[current_root].move_count;
      if (own > 0.5) {
	black++;
      } else {
	white++;
      }
    }
  }

  PrintOwner(&uct_node[current_root], color, owner);

  return black - white;
}


/////////////////////////
//  Owner  //
/////////////////////////
void
OwnerCopy( int *dest )
{
  int pos;
  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    dest[pos] = (int)((double)uct_node[current_root].statistic[pos].colors[my_color] / uct_node[current_root].move_count * 100);
  }
}


///////////////////////////////
//  Criticality  //
///////////////////////////////
void
CopyCriticality( double *dest )
{
  int pos;
  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    dest[pos] = criticality[pos];
  }
}

void
CopyStatistic( statistic_t *dest )
{
  memcpy(dest, statistic, sizeof(statistic_t)* BOARD_MAX); 
}


////////////////////////////////////////////////////////
//  UCT(KGS Clean Up Mode)  //
////////////////////////////////////////////////////////
int
UctSearchGenmoveCleanUp( game_info_t *game, int color )
{
  int pos;
  double finish_time;
  int select_index;
  int max_count;
  double wp;
  int count;
  child_node_t *uct_child;
  thread *handle[THREAD_MAX];

  memset(statistic, 0, sizeof(statistic_t)* board_max); 
  fill_n(criticality_index, board_max, 0); 
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;
  }

  begin_time = ray_clock::now();

  po_info.count = 0;

  current_root = ExpandRoot(game, color);

  if (uct_node[current_root].child_num <= 1) {
    pos = PASS;
    return pos;
  }

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    owner[pos] = 50.0;
  }

  po_info.halt = po_info.num;

  DynamicKomi(game, &uct_node[current_root], color);

  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle[i] = new std::thread(ParallelUctSearch, &t_arg[i]);
  }

  for (int i = 0; i < threads; i++) {
    handle[i]->join();
    delete handle[i];
  }

  uct_child = uct_node[current_root].child;

  select_index = 0;
  max_count = uct_child[0].move_count;

  for (int i = 0; i < uct_node[current_root].child_num; i++){
    if (uct_child[i].move_count > max_count) {
      select_index = i;
      max_count = uct_child[i].move_count;
    }
  }

  finish_time = GetSpendTime(begin_time);

  wp = (double)uct_node[current_root].win / uct_node[current_root].move_count;

  PrintPlayoutInformation(&uct_node[current_root], &po_info, finish_time, 0);
  PrintOwner(&uct_node[current_root], color, owner);

  pos = uct_child[select_index].pos;

  PrintBestSequence(game, uct_node, current_root, color);

  CalculateNextPlayouts(game, color, wp, finish_time);

  count = 0;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];

    if (owner[pos] >= 5 || owner[pos] <= 95) {
      candidates[pos] = true;
      count++;
    } else {
      candidates[pos] = false;
    }
  }

  if (count == 0) pos = PASS;
  else pos = uct_child[select_index].pos;

  if ((double)uct_child[select_index].win / uct_child[select_index].move_count < RESIGN_THRESHOLD) {
    pos = PASS;
  }

  return pos;
}


///////////////////////////////////
//    //
///////////////////////////////////
static void
CorrectDescendentNodes(vector<int> &indexes, int index)
{
  child_node_t *uct_child = uct_node[index].child;
  const int child_num = uct_node[index].child_num;

  indexes.push_back(index);

  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].index != NOT_EXPANDED) {
      CorrectDescendentNodes(indexes, uct_child[i].index);
    }
  }   
}
