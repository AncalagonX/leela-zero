#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "GoBoard.h"
#include "Semeai.h"
#include "UctRating.h"
#include "ZobristHash.h"

using namespace std;

/////////////////////
//         //
/////////////////////

int pure_board_max = PURE_BOARD_MAX;    //     
int pure_board_size = PURE_BOARD_SIZE;  //   
int board_max = BOARD_MAX;              //   
int board_size = BOARD_SIZE;            //  

int board_start = BOARD_START;  // ()
int board_end = BOARD_END;      //  ()

int first_move_candidates;  // 

double komi[S_WHITE + 1];          // 
double dynamic_komi[S_WHITE + 1];  // 
double default_komi = KOMI;        // 

int board_pos_id[BOARD_MAX];  //  

int board_x[BOARD_MAX];  // x  
int board_y[BOARD_MAX];  // y  

unsigned char eye[PAT3_MAX];        // 
unsigned char false_eye[PAT3_MAX];
unsigned char territory[PAT3_MAX];  // 
unsigned char nb4_empty[PAT3_MAX];  // 
eye_condition_t eye_condition[PAT3_MAX];

int border_dis_x[BOARD_MAX];                     // x   
int border_dis_y[BOARD_MAX];                     // y   
int move_dis[PURE_BOARD_SIZE][PURE_BOARD_SIZE];  //   

int onboard_pos[PURE_BOARD_MAX];  //  
int first_move_candidate[PURE_BOARD_MAX]; // 

int corner[4];
int corner_neighbor[4][2];

int cross[4];

bool check_superko = false;  // 

///////////////
//   //
///////////////

// 4
static void InitializeNeighbor( void );

// 
static void InitializeEye( void );

// 
static void InitializeTerritory( void );

// (pos)(string)
// (pos)
static int AddLiberty( string_t *string, const int pos, const int head );

// (pos)(string)
static void RemoveLiberty( game_info_t *game, string_t *string, const int pos );

// (pos)(string)
static void PoRemoveLiberty( game_info_t *game, string_t *string, const int pos, const int color );

// 1
static void MakeString( game_info_t *game, const int pos, const int color );

// 1
static void AddStone( game_info_t *game, int pos, int color, int id );

/// 2
static void ConnectString( game_info_t *game, const int pos, const int color, const int connection, const int id[] );

// 2
static void MergeString( game_info_t *game, string_t *dst, string_t *src[3], const int n );

// 1
static void AddStoneToString( game_info_t *game, string_t *string, const int pos, const int head );

// 
// 
static int RemoveString( game_info_t *game, string_t *string );

// 
// 
static int PoRemoveString( game_info_t *game, string_t *string, const int color );

// ID
static void AddNeighbor( string_t *string, const int id, const int head );

// ID
static void RemoveNeighborString( string_t *string, const int id );

// 
static void CheckBentFourInTheCorner( game_info_t *game );

//  
static bool IsFalseEyeConnection( const game_info_t *game, const int pos, const int color );


//////////////////
//    //
//////////////////
void
SetSuperKo( const bool flag )
{
  check_superko = flag;
}

///////////////////////
//    //
///////////////////////
void
SetBoardSize( const int size )
{
  int i, x, y;

  pure_board_size = size;
  pure_board_max = size * size;
  board_size = size + 2 * OB_SIZE;
  board_max = board_size * board_size;

  board_start = OB_SIZE;
  board_end = (pure_board_size + OB_SIZE - 1);

  i = 0;
  for (y = board_start; y <= board_end; y++) {
    for (x = board_start; x <= board_end; x++) {
      onboard_pos[i++] = POS(x, y);
      board_x[POS(x, y)] = x;
      board_y[POS(x, y)] = y;
    }
  }

  for (y = board_start; y <= board_end; y++) {
    for (x = board_start; x <= (board_start + pure_board_size / 2); x++) {
      border_dis_x[POS(x, y)] = x - (OB_SIZE - 1);
      border_dis_x[POS(board_end + OB_SIZE - x, y)] = x - (OB_SIZE - 1);
      border_dis_y[POS(y, x)] = x - (OB_SIZE - 1);
      border_dis_y[POS(y, board_end + OB_SIZE - x)] = x - (OB_SIZE - 1);
    }
  }

  for (y = 0; y < pure_board_size; y++) {
    for (x = 0; x < pure_board_size; x++) {
      move_dis[x][y] = x + y + ((x > y) ? x : y);
      if (move_dis[x][y] >= MOVE_DISTANCE_MAX) move_dis[x][y] = MOVE_DISTANCE_MAX - 1;
    }
  }

  fill_n(board_pos_id, BOARD_MAX, 0);
  i = 1;
  for (y = board_start; y <= (board_start + pure_board_size / 2); y++) {
    for (x = board_start; x <= y; x++) {
      board_pos_id[POS(x, y)] = i;
      board_pos_id[POS(board_end + OB_SIZE - x, y)] = i;
      board_pos_id[POS(y, x)] = i;
      board_pos_id[POS(y, board_end + OB_SIZE - x)] = i;
      board_pos_id[POS(x, board_end + OB_SIZE - y)] = i;
      board_pos_id[POS(board_end + OB_SIZE - x, board_end + OB_SIZE - y)] = i;
      board_pos_id[POS(board_end + OB_SIZE - y, x)] = i;
      board_pos_id[POS(board_end + OB_SIZE - y, board_end + OB_SIZE - x)] = i;
      i++;
    }
  }

  first_move_candidates = 0;
  for (y = board_start; y <= (board_start + board_end) / 2; y++) {
    for (x = board_end + board_start - y; x <= board_end; x++) {
      first_move_candidate[first_move_candidates++] = POS(x, y);
    }
  }

  corner[0] = POS(board_start, board_start);
  corner[1] = POS(board_start, board_end);
  corner[2] = POS(board_end, board_start);
  corner[3] = POS(board_end, board_end);

  corner_neighbor[0][0] = EAST(POS(board_start, board_start));
  corner_neighbor[0][1] = SOUTH(POS(board_start, board_start));
  corner_neighbor[1][0] = NORTH(POS(board_start, board_end));
  corner_neighbor[1][1] = EAST(POS(board_start, board_end));
  corner_neighbor[2][0] = WEST(POS(board_end, board_start));
  corner_neighbor[2][1] = SOUTH(POS(board_end, board_start));
  corner_neighbor[3][0] = NORTH(POS(board_end, board_end));
  corner_neighbor[3][1] = WEST(POS(board_end, board_end));
}

//////////////////////
//    //
//////////////////////
void
SetKomi( const double new_komi )
{
  default_komi = new_komi;
  komi[0] = dynamic_komi[0] = default_komi;
  komi[S_BLACK] = dynamic_komi[S_BLACK] = default_komi + 1;
  komi[S_WHITE] = dynamic_komi[S_WHITE] = default_komi - 1;
}


////////////////////////////
//    //
////////////////////////////
void
GetNeighbor4( int neighbor4[4], const int pos )
{
  neighbor4[0] = NORTH(pos);
  neighbor4[1] =  WEST(pos);
  neighbor4[2] =  EAST(pos);
  neighbor4[3] = SOUTH(pos);
}

////////////////////////
//    //
////////////////////////
game_info_t *
AllocateGame( void )
{
  game_info_t *game;
  game = new game_info_t();
  memset(game, 0, sizeof(game_info_t));

  return game;
}


////////////////////////
//    //
////////////////////////
void
FreeGame( game_info_t *game )
{
  if (game) delete game;
}


////////////////////////
//    //
////////////////////////
void
InitializeBoard( game_info_t *game )
{
  memset(game->record, 0, sizeof(record_t) * MAX_RECORDS);
  memset(game->pat,    0, sizeof(pattern_t) * board_max);

  fill_n(game->board, board_max, 0);              
  fill_n(game->tactical_features1, board_max, 0);
  fill_n(game->tactical_features2, board_max, 0);
  fill_n(game->update_num,  (int)S_OB, 0);
  fill_n(game->capture_num, (int)S_OB, 0);
  fill(game->update_pos[0],  game->update_pos[S_OB], 0);
  fill(game->capture_pos[0], game->capture_pos[S_OB], 0);
  
  game->current_hash = 0;
  game->previous1_hash = 0;
  game->previous2_hash = 0;
  game->positional_hash = 0;
  game->move_hash = 0;

  SetKomi(default_komi);

  game->moves = 1;

  game->pass_count = 0;

  fill_n(game->candidates, BOARD_MAX, false);

  for (int y = 0; y < board_size; y++){
    for (int x = 0; x < OB_SIZE; x++) {
      game->board[POS(x, y)] = S_OB;
      game->board[POS(y, x)] = S_OB;
      game->board[POS(y, board_size - 1 - x)] = S_OB;
      game->board[POS(board_size - 1 - x, y)] = S_OB;
    }
  }

  for (int y = board_start; y <= board_end; y++) {
    for (int x = board_start; x <= board_end; x++) {
      int pos = POS(x, y);
      game->candidates[pos] = true;
    }
  }

  for (int i = 0; i < MAX_STRING; i++) {
    game->string[i].flag = false;
  }

  ClearPattern(game->pat);

  InitializeNeighbor();
  InitializeEye();
}


//////////////
//    //
//////////////
void
CopyGame( game_info_t *dst, const game_info_t *src )
{
  memcpy(dst->record,             src->record,             sizeof(record_t) * MAX_RECORDS);
  memcpy(dst->prisoner,           src->prisoner,           sizeof(int) * S_MAX);
  memcpy(dst->board,              src->board,              sizeof(char) * board_max);  
  memcpy(dst->pat,                src->pat,                sizeof(pattern_t) * board_max); 
  memcpy(dst->string_id,          src->string_id,          sizeof(int) * STRING_POS_MAX);
  memcpy(dst->string_next,        src->string_next,        sizeof(int) * STRING_POS_MAX);
  memcpy(dst->candidates,         src->candidates,         sizeof(bool) * board_max); 
  memcpy(dst->capture_num,        src->capture_num,        sizeof(int) * S_OB);
  memcpy(dst->update_num,         src->update_num,         sizeof(int) * S_OB);

  fill_n(dst->tactical_features1, board_max, 0);
  fill_n(dst->tactical_features2, board_max, 0);

  for (int i = 0; i < MAX_STRING; i++) {
    if (src->string[i].flag) {
      memcpy(&dst->string[i], &src->string[i], sizeof(string_t));
    } else {
      dst->string[i].flag = false;
    }
  }

  dst->current_hash = src->current_hash;
  dst->previous1_hash = src->previous1_hash;
  dst->previous2_hash = src->previous2_hash;
  dst->positional_hash = src->positional_hash;
  dst->move_hash = src->move_hash;

  dst->pass_count = src->pass_count;

  dst->moves = src->moves;
  dst->ko_move = src->ko_move;
  dst->ko_pos = src->ko_pos;
}



////////////////////
//    //
////////////////////
void
InitializeConst( void )
{
  int i;

  komi[0] = default_komi;
  komi[S_BLACK] = default_komi + 1.0;
  komi[S_WHITE] = default_komi - 1.0;

  i = 0;
  for (int y = board_start; y <= board_end; y++) {
    for (int x = board_start; x <= board_end; x++) {
      onboard_pos[i++] = POS(x, y);
      board_x[POS(x, y)] = x;
      board_y[POS(x, y)] = y;
    }
  }

  for (int y = board_start; y <= board_end; y++) {
    for (int x = board_start; x <= (board_start + pure_board_size / 2); x++) {
      border_dis_x[POS(x, y)] = x - (OB_SIZE - 1);
      border_dis_x[POS(board_end + OB_SIZE - x, y)] = x - (OB_SIZE - 1);
      border_dis_y[POS(y, x)] = x - (OB_SIZE - 1);
      border_dis_y[POS(y, board_end + OB_SIZE - x)] = x - (OB_SIZE - 1);
    }
  }

  for (int y = 0; y < pure_board_size; y++) {
    for (int x = 0; x < pure_board_size; x++) {
      move_dis[x][y] = x + y + ((x > y) ? x : y);
      if (move_dis[x][y] >= MOVE_DISTANCE_MAX) move_dis[x][y] = MOVE_DISTANCE_MAX - 1;
    }
  }

  fill_n(board_pos_id, BOARD_MAX, 0);
  i = 1;
  for (int y = board_start; y <= (board_start + pure_board_size / 2); y++) {
    for (int x = board_start; x <= y; x++) {
      board_pos_id[POS(x, y)] = i;
      board_pos_id[POS(board_end + OB_SIZE - x, y)] = i;
      board_pos_id[POS(y, x)] = i;
      board_pos_id[POS(y, board_end + OB_SIZE - x)] = i;
      board_pos_id[POS(x, board_end + OB_SIZE - y)] = i;
      board_pos_id[POS(board_end + OB_SIZE - x, board_end + OB_SIZE - y)] = i;
      board_pos_id[POS(board_end + OB_SIZE - y, x)] = i;
      board_pos_id[POS(board_end + OB_SIZE - y, board_end + OB_SIZE - x)] = i;
      i++;
    }
  }

  first_move_candidates = 0;
  for (int y = board_start; y <= (board_start + board_end) / 2; y++) {
    for (int x = board_end + board_start - y; x <= board_end; x++) {
      first_move_candidate[first_move_candidates++] = POS(x, y);
    }
  }

  cross[0] = - board_size - 1;
  cross[1] = - board_size + 1;
  cross[2] = board_size - 1;
  cross[3] = board_size + 1;

  corner[0] = POS(board_start, board_start);
  corner[1] = POS(board_start, board_end);
  corner[2] = POS(board_end, board_start);
  corner[3] = POS(board_end, board_end);

  corner_neighbor[0][0] =  EAST(POS(board_start, board_start));
  corner_neighbor[0][1] = SOUTH(POS(board_start, board_start));
  corner_neighbor[1][0] = NORTH(POS(board_start, board_end));
  corner_neighbor[1][1] =  EAST(POS(board_start, board_end));
  corner_neighbor[2][0] =  WEST(POS(board_end, board_start));
  corner_neighbor[2][1] = SOUTH(POS(board_end, board_start));
  corner_neighbor[3][0] = NORTH(POS(board_end, board_end));
  corner_neighbor[3][1] =  WEST(POS(board_end, board_end));

  InitializeNeighbor();
  InitializeEye();
  InitializeTerritory();
}


//////////////////////////////
//    //
//////////////////////////////
static void
InitializeNeighbor( void )
{
  for (int i = 0; i < PAT3_MAX; i++) {
    char empty = 0;

    if (((i >>  2) & 0x3) == S_EMPTY) empty++;
    if (((i >>  6) & 0x3) == S_EMPTY) empty++;
    if (((i >>  8) & 0x3) == S_EMPTY) empty++;
    if (((i >> 12) & 0x3) == S_EMPTY) empty++;

    nb4_empty[i] = empty;
  }
}


////////////////////////////
//    //
////////////////////////////
static void
InitializeEye( void )
{
  unsigned int transp[8], pat3_transp16[16];
  //  12
  //	123
  //	4*5
  //	678
  //  2
  //	O:
  //	X:
  //	+:
  //	#:
  const int eye_pat3[] = {
    // +OO     XOO     +O+     XO+
    // O*O     O*O     O*O     O*O
    // OOO     OOO     OOO     OOO
    0x5554, 0x5556, 0x5544, 0x5546,

    // +OO     XOO     +O+     XO+
    // O*O     O*O     O*O     O*O
    // OO+     OO+     OO+     OO+
    0x1554, 0x1556, 0x1544, 0x1546,

    // +OX     XO+     +OO     OOO
    // O*O     O*O     O*O     O*O
    // OO+     +O+     ###     ###
    0x1564, 0x1146, 0xFD54, 0xFD55,

    // +O#     OO#     XOX     XOX
    // O*#     O*#     O+O     O+O
    // ###     ###     OOO     ###
    0xFF74, 0xFF75, 0x5566, 0xFD66,
  };
  const unsigned int false_eye_pat3[4] = {
    // OOX     OOO     XOO     XO# 
    // O*O     O*O     O*O     O*# 
    // XOO     XOX     ###     ### 
    0x5965, 0x9955, 0xFD56, 0xFF76,
  };

  const unsigned int complete_half_eye[12] = {
    // XOX     OOX     XOX     XOX     XOX
    // O*O     O*O     O*O     O*O     O*O
    // OOO     XOO     +OO     XOO     +O+
    0x5566, 0x5965, 0x5166, 0x5966, 0x1166,
    // +OX     XOX     XOX     XOO     XO+
    // O*O     O*O     O*O     O*O     O*O
    // XO+     XO+     XOX     ###     ###
    0x1964, 0x1966, 0x9966, 0xFD56, 0xFD46,
    // XOX     XO#
    // O*O     O*#
    // ###     ###
    0xFD66, 0xFF76
  };
  const unsigned int half_3_eye[2] = {
    // +O+     XO+
    // O*O     O*O
    // +O+     +O+
    0x1144, 0x1146
  };
  const unsigned int half_2_eye[4] = {
    // +O+     XO+     +OX     +O+
    // O*O     O*O     O*O     O*O
    // +OO     +OO     +OO     ###
    0x5144, 0x5146, 0x5164, 0xFD44,
  };
  const unsigned int half_1_eye[6] = {
    // +O+     XO+     OOX     OOX     +OO
    // O*O     O*O     O*O     O*O     O*O
    // OOO     OOO     +OO     +OO     ###
    0x5544, 0x5564, 0x5145, 0x5165, 0xFD54,
    // +O#
    // O*#
    // ###
    0xFF74,
  };
  const unsigned int complete_one_eye[5] = {
    // OOO     +OO     XOO     OOO     OO#
    // O*O     O*O     O*O     O*O     O*#
    // OOO     OOO     OOO     ###     ###
    0x5555, 0x5554, 0x5556, 0xFD55, 0xFF75,
  };

  fill_n(eye_condition, PAT3_MAX, E_NOT_EYE);
  
  for (int i = 0; i < 12; i++) {
    Pat3Transpose16(complete_half_eye[i], pat3_transp16);
    for (int j = 0; j < 16; j++) {
      eye_condition[pat3_transp16[j]] = E_COMPLETE_HALF_EYE;
    }
  }

  for (int i = 0; i < 2; i++) {
    Pat3Transpose16(half_3_eye[i], pat3_transp16);
    for (int j = 0; j < 16; j++) {
      eye_condition[pat3_transp16[j]] = E_HALF_3_EYE;
    }
  }

  for (int i = 0; i < 4; i++) {
    Pat3Transpose16(half_2_eye[i], pat3_transp16);
    for (int j = 0; j < 16; j++) {
      eye_condition[pat3_transp16[j]] = E_HALF_2_EYE;
    }
  }

  for (int i = 0; i < 6; i++) {
    Pat3Transpose16(half_1_eye[i], pat3_transp16);
    for (int j = 0; j < 16; j++) {
      eye_condition[pat3_transp16[j]] = E_HALF_1_EYE;
    }
  }

  for (int i = 0; i < 5; i++) {
    Pat3Transpose16(complete_one_eye[i], pat3_transp16);
    for (int j = 0; j < 16; j++) {
      eye_condition[pat3_transp16[j]] = E_COMPLETE_ONE_EYE;
    }
  }

  // BBB
  // B*B
  // BBB
  eye[0x5555] = S_BLACK;

  // WWW
  // W*W
  // WWW
  eye[Pat3Reverse(0x5555)] = S_WHITE;

  // +B+
  // B*B
  // +B+
  eye[0x1144] = S_BLACK;

  // +W+
  // W*W
  // +W+
  eye[Pat3Reverse(0x1144)] = S_WHITE;

  for (int i = 0; i < 14; i++) {
    Pat3Transpose8(eye_pat3[i], transp);
    for (int j = 0; j < 8; j++) {
      eye[transp[j]] = S_BLACK;
      eye[Pat3Reverse(transp[j])] = S_WHITE;
    }
  }

  for (int i = 0; i < 4; i++) {
    Pat3Transpose8(false_eye_pat3[i], transp);
    for (int j = 0; j < 8; j++) {
      false_eye[transp[j]] = S_BLACK;
      false_eye[Pat3Reverse(transp[j])] = S_WHITE;
    }
  }

}


/////////////////////////////////////////
//  4  //
/////////////////////////////////////////
static void
InitializeTerritory( void )
{
  for (int i = 0; i < PAT3_MAX; i++) {
    if ((i & 0x1144) == 0x1144) {
      territory[i] = S_BLACK;
    } else if ((i & 0x2288) == 0x2288) {
      territory[i] = S_WHITE;
    }
  }
}


//////////////////
//    //
//////////////////
bool
IsLegal( const game_info_t *game, const int pos, const int color )
{
  // 
  if (game->board[pos] != S_EMPTY) {
    return false;
  }

  // 
  if (nb4_empty[Pat3(game->pat, pos)] == 0 &&
      IsSuicide(game, game->string, color, pos)) {
    return false;
  }

  // 
  if (game->ko_pos == pos &&
      game->ko_move == (game->moves - 1)) {
    return false;
  }

  // 
  if (check_superko &&
      pos != PASS) {
    const int other = FLIP_COLOR(color);
    const string_t *string = game->string;
    const int *string_id = game->string_id;
    const int *string_next = game->string_next;
    unsigned long long hash = game->positional_hash;
    int neighbor4[4], check[4], checked = 0, id, str_pos;
    bool flag;

    GetNeighbor4(neighbor4, pos);

    // 1
    for (int i = 0; i < 4; i++) {
      if (game->board[neighbor4[i]] == other) {
	id = string_id[neighbor4[i]];
	if (string[id].libs == 1) {
	  flag = false;	
	  for (int j = 0; j < checked; j++) {
	    if (check[j] == id) {
	      flag = true;
	    }
	  }
	  if (flag) {
	    continue;
	  }
	  str_pos = string[id].origin;
	  do {
	    hash ^= hash_bit[str_pos][other];
	    str_pos = string_next[str_pos];
	  } while (str_pos != STRING_END);
	}
	check[checked++] = id;
      }
    }

    // poscolor
    hash ^= hash_bit[pos][color];
    
    for (int i = 0; i < game->moves; i++) {
      if (game->record[i].hash == hash) {
	return false;
      }
    }
  }
  
  return true;
}


////////////////////
//    //
////////////////////
static bool
IsFalseEyeConnection( const game_info_t *game, const int pos, const int color )
{
  // +++++XOO#
  // +++++XO+#
  // +++XXXOO#
  // ++XOOXXO#
  // +++O*OO*#
  // #########
  // *,
  // ++++XXXX#
  // +++XXOOO#
  // +++XO+XO#
  // +++XOOO*#
  // #########
  // *.
  //
  // 2
  // .
  // .
  // ++++XXXX#
  // +++XXOOO#
  // +++XOX+O#
  // +++XO+XO#
  // +++XOOO*#
  // #########
  const string_t *string = game->string;
  const int *string_id = game->string_id;
  const char *board = game->board;
  int checked_string[4] = { 0 };
  int string_liberties[4] = { 0 };
  int strings = 0;
  int id, lib, libs = 0, lib_sum = 0;
  int liberty[STRING_LIB_MAX];
  int count;
  bool checked;
  int neighbor4[4], neighbor;
  bool already_checked;
  int other = FLIP_COLOR(color);
  int player_id[4] = {0};
  int player_ids = 0;

  // ID
  GetNeighbor4(neighbor4, pos);
  for (int i = 0; i < 4; i++) {
    checked = false;
    for (int j = 0; j < player_ids; j++) {
      if (player_id[j] == string_id[neighbor4[i]]) {
	checked = true;
      }
    }
    if (!checked) {
      player_id[player_ids++] = string_id[neighbor4[i]];
    }
  }

  // , false
  for (int i = 0; i < 4; i++) {
    if (board[pos + cross[i]] == other) {
      id = string_id[pos + cross[i]];
      if (IsAlreadyCaptured(game, other, id, player_id, player_ids)) {
	return false;
      }
    }
  }

  // 
  // 
  for (int i = 0; i < 4; i++) {
    if (board[neighbor4[i]] == color) {
      id = string_id[neighbor4[i]];
      if (string[id].libs == 2) {
	lib = string[id].lib[0];
	if (lib == pos) lib = string[id].lib[lib];
	if (IsSelfAtari(game, color, lib)) return true;
      }
      already_checked = false;
      for (int j = 0; j < strings; j++) {
	if (checked_string[j] == id) {
	  already_checked = true;
	  break;
	}
      }
      if (already_checked) continue;
      lib = string[id].lib[0];
      count = 0;
      while (lib != LIBERTY_END) {
	if (lib != pos) {
	  checked = false;
	  for (i = 0; i < libs; i++) {
	    if (liberty[i] == lib) {
	      checked = true;
	      break;
	    }
	  }
	  if (!checked) {
	    liberty[libs + count] = lib;
	    count++;
	  }
	}
	lib = string[id].lib[lib];
      }
      libs += count;
      string_liberties[strings] = string[id].libs;
      checked_string[strings++] = id;
    }
  }

  // 
  for (int i = 0; i < strings; i++) {
    lib_sum += string_liberties[i] - 1;
  }

  neighbor = string[checked_string[0]].neighbor[0];
  while (neighbor != NEIGHBOR_END) {
    if (string[neighbor].libs == 1 &&
	string[checked_string[1]].neighbor[neighbor] != 0) {
      return false;
    }
    neighbor = string[checked_string[0]].neighbor[neighbor];
  }

  // false
  if (strings == 1) {
    return false;
  }

  // 2true
  // false
  if (libs == lib_sum) {
    return true;
  } else {
    return false;
  }
}


////////////////////////////////////
//    //
////////////////////////////////////
bool
IsLegalNotEye( game_info_t *game, const int pos, const int color )
{
  const int *string_id = game->string_id;
  const string_t *string = game->string;

  // 
  if (game->board[pos] != S_EMPTY) {
    // 
    game->candidates[pos] = false;

    return false;
  }

  if (game->seki[pos]) {
    return false;
  }

  // 
  if (eye[Pat3(game->pat, pos)] != color ||
      string[string_id[NORTH(pos)]].libs == 1 ||
      string[string_id[ EAST(pos)]].libs == 1 ||
      string[string_id[SOUTH(pos)]].libs == 1 ||
      string[string_id[ WEST(pos)]].libs == 1){

    // 
    if (nb4_empty[Pat3(game->pat, pos)] == 0 &&
	IsSuicide(game, string, color, pos)) {
      return false;
    }

    // 
    if (game->ko_pos == pos &&
	game->ko_move == (game->moves - 1)) {
      return false;
    }

    // 
    if (false_eye[Pat3(game->pat, pos)] == color) {
      if (IsFalseEyeConnection(game, pos, color)) {
	return true;
      } else {
	game->candidates[pos] = false;
	return false;
      }
    }

    return true;
  }

  // 
  game->candidates[pos] = false;

  return false;
}


////////////////////
//    //
////////////////////
bool
IsSuicide( const game_info_t *game, const string_t *string, const int color, const int pos )
{
  const char *board = game->board;
  const int *string_id = game->string_id;
  const int other = FLIP_COLOR(color);
  int neighbor4[4], i;

  GetNeighbor4(neighbor4, pos);

  // 
  // 1
  // 2
  for (i = 0; i < 4; i++) {
    if (board[neighbor4[i]] == other &&
	string[string_id[neighbor4[i]]].libs == 1) {
      return false;
    } else if (board[neighbor4[i]] == color &&
	       string[string_id[neighbor4[i]]].libs > 1) {
      return false;
    }
  }

  return true;
}


////////////////
//    //
////////////////
void
PutStone( game_info_t *game, const int pos, const int color )
{
  const int *string_id = game->string_id;
  char *board = game->board;
  string_t *string = game->string;
  const int other = FLIP_COLOR(color);
  int connection = 0;
  int connect[4] = { 0 };
  int prisoner = 0;
  int neighbor[4];

  // 0
  game->capture_num[color] = 0;

  // 
  game->tactical_features1[pos] = 0;
  game->tactical_features2[pos] = 0;

  game->previous2_hash = game->previous1_hash;
  game->previous1_hash = game->current_hash;

  if (game->ko_move != 0 && game->ko_move == game->moves - 1) {
    game->current_hash ^= hash_bit[game->ko_pos][HASH_KO];
  }

  // 
  if (game->moves < MAX_RECORDS) {
    game->record[game->moves].color = color;
    game->record[game->moves].pos = pos;
    game->move_hash ^= move_bit[game->moves][pos][color];
  }

  // 
  if (pos == PASS) {
    if (game->moves < MAX_RECORDS) {
      game->record[game->moves].hash = game->positional_hash;
    }
    game->current_hash ^= hash_bit[game->pass_count++][HASH_PASS];
    if (game->pass_count >= BOARD_MAX) { 
      game->pass_count = 0;
    }
    game->moves++;
    return;
  }

  // 
  board[pos] = (char)color;

  // 
  game->candidates[pos] = false;

  game->current_hash ^= hash_bit[pos][color];
  game->positional_hash ^= hash_bit[pos][color];

  // (MD5)
  UpdatePatternStone(game->pat, color, pos);

  // 
  GetNeighbor4(neighbor, pos);

  // 
  // , 1, 
  // , 1, 0
  for (int i = 0; i < 4; i++) {
    if (board[neighbor[i]] == color) {
      RemoveLiberty(game, &string[string_id[neighbor[i]]], pos);
      connect[connection++] = string_id[neighbor[i]];
    } else if (board[neighbor[i]] == other) {
      RemoveLiberty(game, &string[string_id[neighbor[i]]], pos);
      if (string[string_id[neighbor[i]]].libs == 0) {
	prisoner += RemoveString(game, &string[string_id[neighbor[i]]]);
      }
    }
  }

  // 
  game->prisoner[color] += prisoner;

  // , , 
  // 1, 
  // 2, , 
  if (connection == 0) {
    MakeString(game, pos, color);
    if (prisoner == 1 &&
	string[string_id[pos]].libs == 1) {
      game->ko_move = game->moves;
      game->ko_pos = string[string_id[pos]].lib[0];
      game->current_hash ^= hash_bit[game->ko_pos][HASH_KO];
    }
  } else if (connection == 1) {
    AddStone(game, pos, color, connect[0]);
  } else {
    ConnectString(game, pos, color, connection, connect);
  }

  // 
  if (game->moves < MAX_RECORDS) {
    game->record[game->moves].hash = game->positional_hash;
  }
  
  // 1
  game->moves++;
}


////////////////
//    //
////////////////
void
PoPutStone( game_info_t *game, const int pos, const int color )
{
  const int *string_id = game->string_id;
  char *board = game->board;
  string_t *string = game->string;
  const int other = FLIP_COLOR(color);
  int connection = 0;
  int connect[4] = { 0 };
  int prisoner = 0;
  int neighbor[4];

  // 0
  game->capture_num[color] = 0;

  // 
  if (game->moves < MAX_RECORDS) {
    game->record[game->moves].color = color;
    game->record[game->moves].pos = pos;
  }

  // 
  if (pos == PASS) {
    game->moves++;
    return;
  }

  // 
  board[pos] = (char)color;

  // 
  game->candidates[pos] = false;

  // 
  game->tactical_features1[pos] = 0;
  game->tactical_features2[pos] = 0;

  // 0
  game->sum_rate[0] -= game->rate[0][pos];
  game->sum_rate_row[0][board_y[pos]] -= game->rate[0][pos];
  game->rate[0][pos] = 0;
  game->sum_rate[1] -= game->rate[1][pos];
  game->sum_rate_row[1][board_y[pos]] -= game->rate[1][pos];
  game->rate[1][pos] = 0;

  // (MD2)  
  UpdateMD2Stone(game->pat, color, pos);

  // 
  GetNeighbor4(neighbor, pos);

  // 
  // , 1, 
  // , 1, 0  
  for (int i = 0; i < 4; i++) {
    if (board[neighbor[i]] == color) {
      PoRemoveLiberty(game, &string[string_id[neighbor[i]]], pos, color);
      connect[connection++] = string_id[neighbor[i]];
    } else if (board[neighbor[i]] == other) {
      PoRemoveLiberty(game, &string[string_id[neighbor[i]]], pos, color);
      if (string[string_id[neighbor[i]]].libs == 0) {
	prisoner += PoRemoveString(game, &string[string_id[neighbor[i]]], color);
      }
    }
  }

  // 
  game->prisoner[color] += prisoner;

  // , , 
  // 1, 
  // 2, ,   
  if (connection == 0) {
    MakeString(game, pos, color);
    if (prisoner == 1 &&
	string[string_id[pos]].libs == 1) {
      game->ko_move = game->moves;
      game->ko_pos = string[string_id[pos]].lib[0];
    }
  } else if (connection == 1) {
    AddStone(game, pos, color, connect[0]);
  } else {
    ConnectString(game, pos, color, connection, connect);
  }

  // 
  game->moves++;
}


//////////////////////
//    //
//////////////////////
static void
MakeString( game_info_t *game, const int pos, const int color )
{
  string_t *string = game->string;
  string_t *new_string;
  const char *board = game->board;
  int *string_id = game->string_id;
  int id = 1;
  int lib_add = 0;
  int other = FLIP_COLOR(color);
  int neighbor, neighbor4[4], i;

  // 
  while (string[id].flag) { id++; }

  // 
  new_string = &game->string[id];

  // 
  fill_n(new_string->lib, STRING_LIB_MAX, 0);
  fill_n(new_string->neighbor, MAX_NEIGHBOR, 0);
  new_string->lib[0] = LIBERTY_END;
  new_string->neighbor[0] = NEIGHBOR_END;
  new_string->libs = 0;
  new_string->color = (char)color;
  new_string->origin = pos;
  new_string->size = 1;
  new_string->neighbors = 0;
  game->string_id[pos] = id;
  game->string_next[pos] = STRING_END;

  // 
  GetNeighbor4(neighbor4, pos);

  // 
  // , 
  // , 
  for (i = 0; i < 4; i++) {
    if (board[neighbor4[i]] == S_EMPTY) {
      lib_add = AddLiberty(new_string, neighbor4[i], lib_add);
    } else if (board[neighbor4[i]] == other) {
      neighbor = string_id[neighbor4[i]];
      AddNeighbor(&string[neighbor], id, 0);
      AddNeighbor(&string[id], neighbor, 0);
    }
  }

  // 
  new_string->flag = true;
}


///////////////////////////
//  1  //
///////////////////////////
static void
AddStoneToString( game_info_t *game, string_t *string, const int pos, const int head )
// game_info_t *game :  
// string_t *string  : 
// int pos         : 
// int head        : 
{
  int *string_next = game->string_next;
  int str_pos;

  if (pos == STRING_END) return;

  // 
  // 
  if (string->origin > pos) {
    string_next[pos] = string->origin;
    string->origin = pos;
  } else {
    if (head != 0) {
      str_pos = head;
    } else {
      str_pos = string->origin;
    }
    while (string_next[str_pos] < pos){
      str_pos = string_next[str_pos];
    }
    string_next[pos] = string_next[str_pos];
    string_next[str_pos] = pos;
  }
  string->size++;
}


////////////////////////
//    //
////////////////////////
static void
AddStone( game_info_t *game, const int pos, const int color, const int id )
// game_info_t *game : 
// int pos           : 
// int color         : 
// int id            : ID
{
  string_t *string = game->string;
  string_t *add_str;
  char *board = game->board;
  int *string_id = game->string_id;
  int lib_add = 0;
  int other = FLIP_COLOR(color);
  int neighbor, neighbor4[4], i;

  // ID
  string_id[pos] = id;

  // 
  add_str = &string[id];

  // 
  AddStoneToString(game, add_str, pos, 0);

  // 
  GetNeighbor4(neighbor4, pos);

  // 
  // 
  for (i = 0; i < 4; i++) {
    if (board[neighbor4[i]] == S_EMPTY) {
      lib_add = AddLiberty(add_str, neighbor4[i], lib_add);
    } else if (board[neighbor4[i]] == other) {
      neighbor = string_id[neighbor4[i]];
      AddNeighbor(&string[neighbor], id, 0);
      AddNeighbor(&string[id], neighbor, 0);
    }
  }
}


//////////////////////////
//    //
//////////////////////////
static void
ConnectString( game_info_t *game, const int pos, const int color, const int connection, const int id[] )
// game_info_t *game : 
// int pos           : 
// int color         : 
// int connection    : 
// int id[]          : ID
{
  int min = id[0];
  string_t *string = game->string;
  string_t *str[3];
  int connections = 0;
  bool flag = true;

  // 
  for (int i = 1; i < connection; i++) {
    flag = true;
    for (int j = 0; j < i; j++) {
      if (id[j] == id[i]) {
	flag = false;
	break;
      }
    }
    if (flag) {
      if (min > id[i]) {
	str[connections] = &string[min];
	min = id[i];
      } else {
	str[connections] = &string[id[i]];
      }
      connections++;
    }
  }

  // 
  AddStone(game, pos, color, min);

  // 
  if (connections > 0) {
    MergeString(game, &game->string[min], str, connections);
  }
}


////////////////
//    //
////////////////
static void
MergeString( game_info_t *game, string_t *dst, string_t *src[3], const int n )
// game_info_t *game : 
// string_t *dst     : 
// string_t *src[3]  : (3)
// int n             : 
{
  int tmp, pos, prev, neighbor;
  int *string_next = game->string_next;
  int *string_id = game->string_id;
  int id = string_id[dst->origin], rm_id;
  string_t *string = game->string;

  for (int i = 0; i < n; i++) {
    // ID
    rm_id = string_id[src[i]->origin];

    // 
    prev = 0;
    pos = src[i]->lib[0];
    while (pos != LIBERTY_END) {
      prev = AddLiberty(dst, pos, prev);
      pos = src[i]->lib[pos];
    }

    // ID
    prev = 0;
    pos = src[i]->origin;
    while (pos != STRING_END) {
      string_id[pos] = id;
      tmp = string_next[pos];
      AddStoneToString(game, dst, pos, prev);
      prev = pos;
      pos = tmp;
    }

    // 
    prev = 0;
    neighbor = src[i]->neighbor[0];
    while (neighbor != NEIGHBOR_END) {
      RemoveNeighborString(&string[neighbor], rm_id);
      AddNeighbor(dst, neighbor, prev);
      AddNeighbor(&string[neighbor], id, prev);
      prev = neighbor;
      neighbor = src[i]->neighbor[neighbor];
    }

    // 
    src[i]->flag = false;
  }
}


////////////////////
//    //
////////////////////
static int
AddLiberty( string_t *string, const int pos, const int head )
// string_t *string : 
// int pos        : 
// int head       : 
{
  int lib;

  // 
  if (string->lib[pos] != 0) return pos;

  // 
  lib = head;

  // 
  while (string->lib[lib] < pos) {
    lib = string->lib[lib];
  }

  // 
  string->lib[pos] = string->lib[lib];
  string->lib[lib] = (short)pos;

  // 1
  string->libs++;

  // 
  return pos;
}


////////////////////
//    //
////////////////////
static void
RemoveLiberty( game_info_t *game, string_t *string, const int pos )
// game_info_t *game : 
// string_t *string  : 
// int pos         : 
{
  int lib = 0;

  // 
  if (string->lib[pos] == 0) return;

  // 
  while (string->lib[lib] != pos) {
    lib = string->lib[lib];
  }

  // 
  string->lib[lib] = string->lib[string->lib[lib]];
  string->lib[pos] = (short)0;

  // 1
  string->libs--;

  // 1, 
  if (string->libs == 1) {
    game->candidates[string->lib[0]] = true;
  }
}


//////////////////////
//      //
// () //
//////////////////////
static void
PoRemoveLiberty( game_info_t *game, string_t *string, const int pos, const int color )
// game_info_t *game : 
// string_t *string  : 
// int pos         : 
// int color       : 
{
  int lib = 0;

  // 
  if (string->lib[pos] == 0) return;

  // 
  while (string->lib[lib] != pos) {
    lib = string->lib[lib];
  }

  // 
  string->lib[lib] = string->lib[string->lib[lib]];
  string->lib[pos] = 0;

  // 1
  string->libs--;

  // 
  // 1, , 
  // 2, 
  if (string->libs == 1) {
    game->candidates[string->lib[0]] = true;
    game->update_pos[color][game->update_num[color]++] = string->lib[0];
    game->seki[string->lib[0]] = false;
  }
}


////////////////
//    //
////////////////
static int
RemoveString( game_info_t *game, string_t *string )
// game_info_t *game : 
// string_t *string  : 
{
  string_t *str = game->string;
  int *string_next = game->string_next;
  int *string_id = game->string_id;
  int pos = string->origin, next;
  char *board = game->board;
  bool *candidates = game->candidates;
  int neighbor, rm_id = string_id[string->origin];
  int removed_color = board[pos];

  do {
    // 
    board[pos] = S_EMPTY;

    // 
    candidates[pos] = true;

    // 
    UpdatePatternEmpty(game->pat, pos);

    game->current_hash ^= hash_bit[pos][removed_color];
    game->positional_hash ^= hash_bit[pos][removed_color];

    // 
    // 
    if (str[string_id[NORTH(pos)]].flag) AddLiberty(&str[string_id[NORTH(pos)]], pos, 0);
    if (str[string_id[ WEST(pos)]].flag) AddLiberty(&str[string_id[ WEST(pos)]], pos, 0);
    if (str[string_id[ EAST(pos)]].flag) AddLiberty(&str[string_id[ EAST(pos)]], pos, 0);
    if (str[string_id[SOUTH(pos)]].flag) AddLiberty(&str[string_id[SOUTH(pos)]], pos, 0);

    // 
    next = string_next[pos];

    // , 
    // ID
    string_next[pos] = 0;
    string_id[pos] = 0;

    // 
    pos = next;
  } while (pos != STRING_END);

  // 
  neighbor = string->neighbor[0];
  while (neighbor != NEIGHBOR_END) {
    RemoveNeighborString(&str[neighbor], rm_id);
    neighbor = string->neighbor[neighbor];
  }

  // 
  string->flag = false;

  // 
  return string->size;
}


////////////////
//    //
////////////////
static int
PoRemoveString( game_info_t *game, string_t *string, const int color )
// game_info_t *game : 
// string_t *string  : 
// int color       : ()
{
  string_t *str = game->string;
  int *string_next = game->string_next;
  int *string_id = game->string_id;
  int pos = string->origin, next;
  char *board = game->board;
  bool *candidates = game->candidates;
  int neighbor, rm_id = string_id[string->origin];
  int *capture_pos = game->capture_pos[color];
  int *capture_num = &game->capture_num[color];
  int *update_pos = game->update_pos[color];
  int *update_num = &game->update_num[color];
  int lib;
  
  // 
  neighbor = string->neighbor[0];
  while (neighbor != NEIGHBOR_END) {
    if (str[neighbor].libs < 3) {
      lib = str[neighbor].lib[0];
      while (lib != LIBERTY_END) {
	update_pos[(*update_num)++] = lib;
	game->seki[lib] = false;
	lib = str[neighbor].lib[lib];
      }
    }
    neighbor = string->neighbor[neighbor];
  }
  
  do {
    // 
    board[pos] = S_EMPTY;
    // 
    candidates[pos] = true;

    // 
    capture_pos[(*capture_num)++] = pos;

    // 3x3
    UpdateMD2Empty(game->pat, pos);
    
    // 
    // 
    if (str[string_id[NORTH(pos)]].flag) AddLiberty(&str[string_id[NORTH(pos)]], pos, 0);
    if (str[string_id[ WEST(pos)]].flag) AddLiberty(&str[string_id[ WEST(pos)]], pos, 0);
    if (str[string_id[ EAST(pos)]].flag) AddLiberty(&str[string_id[ EAST(pos)]], pos, 0);
    if (str[string_id[SOUTH(pos)]].flag) AddLiberty(&str[string_id[SOUTH(pos)]], pos, 0);

    // 
    next = string_next[pos];

    // , 
    // ID
    string_next[pos] = 0;
    string_id[pos] = 0;

    // 
    pos = next;
  } while (pos != STRING_END);

  // 
  neighbor = string->neighbor[0];
  while (neighbor != NEIGHBOR_END) {
    RemoveNeighborString(&str[neighbor], rm_id);
    neighbor = string->neighbor[neighbor];
  }

  // 
  string->flag = false;

  // 
  return string->size;
}


////////////////////////////////////
//  ID()  //
////////////////////////////////////
static void
AddNeighbor( string_t *string, const int id, const int head )
// string_t *string : 
// int id         : ID
// int head       : 
{
  int neighbor = 0;

  // 
  if (string->neighbor[id] != 0) return;

  // 
  neighbor = head;

  // 
  while (string->neighbor[neighbor] < id) {
    neighbor = string->neighbor[neighbor];
  }

  // ID
  string->neighbor[id] = string->neighbor[neighbor];
  string->neighbor[neighbor] = (short)id;

  // 1
  string->neighbors++;
}


//////////////////////////
//  ID  //
//////////////////////////
static void
RemoveNeighborString( string_t *string, const int id )
// string_t *string : ID
// int id         : ID
{
  int neighbor = 0;

  // 
  if (string->neighbor[id] == 0) return;

  // ID
  while (string->neighbor[neighbor] != id) {
    neighbor = string->neighbor[neighbor];
  }

  // ID
  string->neighbor[neighbor] = string->neighbor[string->neighbor[neighbor]];
  string->neighbor[id] = 0;

  // 1
  string->neighbors--;
}


///////////////////////////
//    //
///////////////////////////
static void
CheckBentFourInTheCorner( game_info_t *game )
{
  char *board = game->board;
  const string_t *string = game->string;
  const int *string_id = game->string_id;
  const int *string_next = game->string_next;
  int pos;
  int i;
  int id;
  int neighbor;
  int color;
  int lib1, lib2;
  int neighbor_lib1, neighbor_lib2;

  // 
  // 
  for (i = 0; i < 4; i++) {
    id = string_id[corner[i]];
    if (string[id].size == 3 &&
	string[id].libs == 2 &&
	string[id].neighbors == 1) {
      color = string[id].color;
      lib1 = string[id].lib[0];
      lib2 = string[id].lib[lib1];
      if ((board[corner_neighbor[i][0]] == S_EMPTY ||
	   board[corner_neighbor[i][0]] == color) &&
	  (board[corner_neighbor[i][1]] == S_EMPTY ||
	   board[corner_neighbor[i][1]] == color)) {
	neighbor = string[id].neighbor[0];
	if (string[neighbor].libs == 2 &&
	    string[neighbor].size > 6) {
	  // 
	  neighbor_lib1 = string[neighbor].lib[0];
	  neighbor_lib2 = string[neighbor].lib[neighbor_lib1];
	  if ((neighbor_lib1 == lib1 && neighbor_lib2 == lib2) ||
	      (neighbor_lib1 == lib2 && neighbor_lib2 == lib1)) {
	    pos = string[neighbor].origin;
	    while (pos != STRING_END) {
	      board[pos] = (char)color;
	      pos = string_next[pos];
	    }
	    pos = string[neighbor].lib[0];
	    board[pos] = (char)color;
	    pos = string[neighbor].lib[pos];
	    board[pos] = (char)color;
	  }
	}
      }
    }
  }
}


////////////////
//    //
////////////////
int
CalculateScore( game_info_t *game )
// game_info_t *game : 
{
  const char *board = game->board;
  int i;
  int pos;
  int color;
  int scores[S_MAX] = { 0 };

  // 
  CheckBentFourInTheCorner(game);

  // 
  for (i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    color = board[pos];
    if (color == S_EMPTY) color = territory[Pat3(game->pat, pos)];
    scores[color]++;
  }

  //  ()
  return(scores[S_BLACK] - scores[S_WHITE]);
}
