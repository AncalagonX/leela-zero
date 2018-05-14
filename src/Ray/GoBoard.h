#ifndef _GO_BOARD_H_
#define _GO_BOARD_H_

#include "Pattern.h"

////////////////
//        //
////////////////

const int PURE_BOARD_SIZE = 19;  // 

const int OB_SIZE = 5; // 
const int BOARD_SIZE = (PURE_BOARD_SIZE + OB_SIZE + OB_SIZE); // 

const int PURE_BOARD_MAX = (PURE_BOARD_SIZE * PURE_BOARD_SIZE); //  
const int BOARD_MAX = (BOARD_SIZE * BOARD_SIZE);                // 

const int MAX_STRING = (PURE_BOARD_MAX * 4 / 5); //  
const int MAX_NEIGHBOR = MAX_STRING;             // 

const int BOARD_START = OB_SIZE;                        //   
const int BOARD_END = (PURE_BOARD_SIZE + OB_SIZE - 1);  //   

const int STRING_LIB_MAX = (BOARD_SIZE * (PURE_BOARD_SIZE + OB_SIZE));  // 1
const int STRING_POS_MAX = (BOARD_SIZE * (PURE_BOARD_SIZE + OB_SIZE));  // 

const int STRING_END = (STRING_POS_MAX - 1); // 
const int NEIGHBOR_END = (MAX_NEIGHBOR - 1);  // 
const int LIBERTY_END = (STRING_LIB_MAX - 1); // 

const int MAX_RECORDS = (PURE_BOARD_MAX * 3); //  
const int MAX_MOVES = (MAX_RECORDS - 1);      // 

const int PASS = 0;     // 
const int RESIGN = -1;  // 

const double KOMI = 6.5; // 

//////////////////
//    //
//////////////////
#define POS(x, y) ((x) + (y) * board_size)  // (x, y)
#define X(pos)        ((pos) % board_size)  // posx
#define Y(pos)        ((pos) / board_size)  // posy

#define CORRECT_X(pos) ((pos) % board_size - OB_SIZE + 1)  // x
#define CORRECT_Y(pos) ((pos) / board_size - OB_SIZE + 1)  // y

#define NORTH(pos) ((pos) - board_size)  // pos
#define  WEST(pos) ((pos) - 1)           // pos
#define  EAST(pos) ((pos) + 1)           // pos
#define SOUTH(pos) ((pos) + board_size)  // pos

#define FLIP_COLOR(col) ((col) ^ 0x3) // 

#define DX(pos1, pos2)  (abs(board_x[(pos1)] - board_x[(pos2)]))     // x
#define DY(pos1, pos2)  (abs(board_y[(pos1)] - board_y[(pos2)]))     // y
#define DIS(pos1, pos2) (move_dis[DX(pos1, pos2)][DY(pos1, pos2)])   // 


enum stone {
  S_EMPTY,  // 
  S_BLACK,  // 
  S_WHITE,  // 
  S_OB,     // 
  S_MAX     // 
};

enum eye_condition_t : unsigned char {
  E_NOT_EYE,           // 
  E_COMPLETE_HALF_EYE, // (81)
  E_HALF_3_EYE,        // , 31
  E_HALF_2_EYE,        // , 21
  E_HALF_1_EYE,        // , 11
  E_COMPLETE_ONE_EYE,  // 1
  E_MAX,
};

// 
struct record_t {
  int color;                // 
  int pos;                  // 
  unsigned long long hash;  // 
};

//  (19x19 : 1987bytes)
struct string_t {
  char color;                    // 
  int libs;                      // 
  short lib[STRING_LIB_MAX];     // 
  int neighbors;                 // 
  short neighbor[MAX_NEIGHBOR];  // 
  int origin;                    // 
  int size;                      // 
  bool flag;                     // 
};


// 
struct game_info_t {
  record_t record[MAX_RECORDS];  // 
  int moves;                        // 
  int prisoner[S_MAX];              // 
  int ko_pos;                       // 
  int ko_move;                      // 

  unsigned long long current_hash;     // 
  unsigned long long previous1_hash;   // 1
  unsigned long long previous2_hash;   // 2
  unsigned long long positional_hash;  // ()
  unsigned long long move_hash;        // ()
  
  char board[BOARD_MAX];            //  

  int pass_count;                   // 

  pattern_t pat[BOARD_MAX];    //  

  string_t string[MAX_STRING];        // (19x19 : 573,845bytes)
  int string_id[STRING_POS_MAX];    // ID
  int string_next[STRING_POS_MAX];  // 

  bool candidates[BOARD_MAX];  //  
  bool seki[BOARD_MAX];
  
  unsigned int tactical_features1[BOARD_MAX];  //  
  unsigned int tactical_features2[BOARD_MAX];  //  

  int capture_num[S_OB];                   // 
  int capture_pos[S_OB][PURE_BOARD_MAX];   //  

  int update_num[S_OB];                    // 
  int update_pos[S_OB][PURE_BOARD_MAX];    //  

  long long rate[2][BOARD_MAX];           //  
  long long sum_rate_row[2][BOARD_SIZE];  //   
  long long sum_rate[2];                  // 
};


////////////////
//        //
////////////////

// 
extern int pure_board_size;

// 
extern int pure_board_max;

// ()
extern int board_size;

// ()
extern int board_max;

// ()
extern int board_start;

// ()
extern int board_end;

// 
extern int first_move_candidates;

// 
extern double komi[S_OB];

// Dynamic Komi
extern double dynamic_komi[S_OB];

// ID
extern int board_pos_id[BOARD_MAX];  

// x
extern int board_x[BOARD_MAX];  

//  y
extern int board_y[BOARD_MAX];  

// 
extern unsigned char eye[PAT3_MAX];

// 
extern unsigned char territory[PAT3_MAX];

// 4
extern unsigned char nb4_empty[PAT3_MAX];

// 
extern eye_condition_t eye_condition[PAT3_MAX];

// x
extern int border_dis_x[BOARD_MAX]; 

// y
extern int border_dis_y[BOARD_MAX]; 

// 
extern int move_dis[PURE_BOARD_SIZE][PURE_BOARD_SIZE];

// 
extern int onboard_pos[PURE_BOARD_MAX];

// 
extern int first_move_candidate[PURE_BOARD_MAX];

//////////////
//      //
//////////////

// 
void SetSuperKo( const bool flag );

// 
void SetBoardSize( const int size );

// 
game_info_t *AllocateGame( void );

// 
void FreeGame( game_info_t *game );

// 
void CopyGame( game_info_t *dst, const game_info_t *src );

// 
void InitializeConst( void );

// 
void InitializeBoard( game_info_t *game );

// 
// true
bool IsLegal( const game_info_t *game, const int pos, const int color );

// 
// true
bool IsLegalNotEye( game_info_t *game, const int pos, const int color );

// 
// true
bool IsSuicide( const game_info_t *game, const string_t *string, const int color, const int pos );

// 
void PutStone( game_info_t *game, const int pos, const int color );

// ()
void PoPutStone( game_info_t *game, const int pos, const int color );

// 
int CalculateScore( game_info_t *game );

// 
void SetKomi( const double new_komi );

// 
void GetNeighbor4( int neighbor4[4], const int pos );

#endif
