#ifndef _SEARCH_BOARD_H_
#define _SEARCH_BOARD_H_

#include "GoBoard.h"

//////////////
//    //
//////////////
struct undo_record_t {
  int stone[4][PURE_BOARD_MAX];  // 
  int stones[4];                 // 
  int strings_id[4];             // ID
  int strings;                   // 
  char string_color[4];          // 
  int ko_move_record;            // 
  int ko_pos_record;             // 
};
struct search_game_info_t{
  explicit search_game_info_t(const game_info_t *src);

  record_t record[MAX_RECORDS];               // 
  int moves;                                  // 
  int prisoner[S_MAX];                        // 
  int ko_pos;                                 // 
  int ko_move;                                // 
  char board[BOARD_MAX];                      // 
  pattern_t pat[BOARD_MAX];                   // 
  string_t string[MAX_STRING];                // 
  int string_id[STRING_POS_MAX];              // ID
  int string_next[STRING_POS_MAX];            // 
  bool candidates[BOARD_MAX];                 // 
  undo_record_t undo[MAX_RECORDS];            // 
};

////////////
//    //
////////////

// 
void PutStoneForSearch( search_game_info_t *game, const int pos, const int color );

// 
bool IsLegalForSearch( const search_game_info_t *game, const int pos, const int color );

// 
void SearchBoardTest( void );

// 1
void Undo( search_game_info_t *game );

// 
void PrintSearchBoard( const search_game_info_t *game ); 

#endif
