#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "SearchBoard.h"

using namespace std;

////////////
//    //
////////////

//   
static int AddLiberty( string_t *string, const int pos, const int head );

//  ID
static void AddNeighbor( string_t *string, const int id );

//  
static void AddStoneToString( search_game_info_t *game, string_t *string, const int pos);
//    
static void AddStone( search_game_info_t *game, const int pos, const int color, const int id );

//  
static void ConnectString( search_game_info_t *game, const int pos, const int color, const int connection, const int id[] );

//  
static bool IsSuicide( const search_game_info_t *game, const string_t *string, const int color, const int pos );

//  
static void MakeString( search_game_info_t *game, const int pos, const int color );

//  
static void MergeLiberty( string_t *dst, string_t *src );

//  ID
static void MergeNeighbor( string_t *string, string_t *dst, string_t *src, const int id, const int rm_id );

//   
static void MergeStones( search_game_info_t *game, const int id, const int rm_id );

//   
static void MergeString( search_game_info_t *game, string_t *dst, string_t *src[3], const int n );

//  
static void RecordString( search_game_info_t *game, int id );

//  1
static void RemoveLiberty( search_game_info_t *game, string_t *string, const int pos );

//  ID
static void RemoveNeighborString( string_t *string, const int id );

//  
static int RemoveString( search_game_info_t *game, string_t *string );

//   
static void RestoreChain( search_game_info_t *game, const int id, const int stone[], const int stones, const int color );


///////////////////
//    //
///////////////////
search_game_info_t::search_game_info_t( const game_info_t *src )
{
  memcpy(record,      src->record,      sizeof(record_t) * MAX_RECORDS);
  memcpy(prisoner,    src->prisoner,    sizeof(int) * S_MAX);
  memcpy(board,       src->board,       sizeof(char) * BOARD_MAX);
  memcpy(pat,         src->pat,         sizeof(pattern_t) * BOARD_MAX);
  memcpy(string_id,   src->string_id,   sizeof(int) * STRING_POS_MAX);
  memcpy(string_next, src->string_next, sizeof(int) * STRING_POS_MAX);
  memcpy(candidates,  src->candidates, sizeof(bool) * BOARD_MAX);

  for (int i = 0; i < MAX_STRING; i++) {
    if (src->string[i].flag) {
      memcpy(&string[i], &src->string[i], sizeof(string_t));
    } else {
      string[i].flag = false;
    }
  }

  moves = src->moves;
  ko_move = src->ko_move;
  ko_pos = src->ko_pos;

  memset(&undo[moves], 0, sizeof(undo_record_t));
  undo[moves].ko_move_record = ko_move;
  undo[moves].ko_pos_record = ko_pos;
}


//////////////////
//    //
//////////////////
bool
IsLegalForSearch( const search_game_info_t *game, const int pos, const int color )
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

  return true;
}


///////////////////////////
//    //
///////////////////////////
static void
RecordString( search_game_info_t *game, int id )
{
  const int moves = game->moves;  
  const string_t *string = game->string;
  const int *string_next = game->string_next;
  undo_record_t* rec = &game->undo[moves];
  int pos, i = 0;

  pos = string[id].origin;
  while (pos != STRING_END) {
    rec->stone[rec->strings][i++] = pos;
    pos = string_next[pos];
  }

  rec->string_color[rec->strings] = string[id].color;
  rec->stones[rec->strings] = string[id].size;
  rec->strings_id[rec->strings] = id;
  rec->strings++;
}


////////////////
//    //
////////////////
void
PutStoneForSearch( search_game_info_t *game, const int pos, const int color )
{
  int *string_id = game->string_id;
  char *board = game->board;
  string_t *string = game->string;
  int other = FLIP_COLOR(color);
  int connection = 0;
  int connect[4] = { 0 };
  int prisoner = 0;
  int neighbor[4];

  // 
  if (game->moves < MAX_RECORDS) {
    game->record[game->moves].color = color;
    game->record[game->moves].pos = pos;
    game->undo[game->moves].strings = 0;
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

  // (MD5)
  UpdateMD2Stone(game->pat, color, pos);

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
	RecordString(game, string_id[neighbor[i]]);
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
    }
  } else if (connection == 1) {
    RecordString(game, connect[0]);
    AddStone(game, pos, color, connect[0]);
  } else {
    ConnectString(game, pos, color, connection, connect);
  }


  // 1
  game->moves++;
  memset(&game->undo[game->moves], 0, sizeof(undo_record_t));
  game->undo[game->moves].ko_move_record = game->ko_move;
  game->undo[game->moves].ko_pos_record = game->ko_pos;

}


///////////////////
//    //
///////////////////
static int
AddLiberty( string_t *string, const int pos, const int head )
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


/////////////////////////////
//  ID  //
/////////////////////////////
static void
AddNeighbor( string_t *string, const int id )
{
  int neighbor = 0;

  // 
  if (string->neighbor[id] != 0) return;

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


///////////////////
//    //
///////////////////
static void
AddStoneToString( search_game_info_t *game, string_t *string, const int pos )
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
    str_pos = string->origin;
    while (string_next[str_pos] < pos) {
      str_pos = string_next[str_pos];
    }
    string_next[pos] = string_next[str_pos];
    string_next[str_pos] = pos;
  }
  string->size++;
}


////////////////
//    //
////////////////
static void
AddStone( search_game_info_t *game, const int pos, const int color, const int id )
{
  const char *board = game->board;
  string_t *string = game->string;
  string_t *add_str;
  int *string_id = game->string_id;
  int lib_add = 0;
  int other = FLIP_COLOR(color);
  int neighbor, neighbor4[4];

  // ID
  string_id[pos] = id;

  // 
  add_str = &string[id];

  // 
  AddStoneToString(game, add_str, pos);

  // 
  GetNeighbor4(neighbor4, pos);

  // 
  // 
  for (int i = 0; i < 4; i++) {
    if (board[neighbor4[i]] == S_EMPTY) {
      lib_add = AddLiberty(add_str, neighbor4[i], lib_add);
    } else if (board[neighbor4[i]] == other) {
      neighbor = string_id[neighbor4[i]];
      AddNeighbor(&string[neighbor], id);
      AddNeighbor(&string[id], neighbor);
    }
  }
}


////////////////
//    //
////////////////
static void
ConnectString( search_game_info_t *game, const int pos, const int color, const int connection, const int id[] )
{
  int min = id[0];
  int ids[4];
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
      ids[connections] = id[i];
      connections++;
    }
  }

  ids[connections] = id[0];

  for (int i = 0; i <= connections; i++) {
    RecordString(game, ids[i]);
  }

  // 
  AddStone(game, pos, color, min);

  // 
  if (connections > 0) {
    MergeString(game, &game->string[min], str, connections);
  }
}


////////////////////
//    //
///////////////////
static bool
IsSuicide( const search_game_info_t *game, const string_t *string, const int color, const int pos )
{
  const char *board = game->board;
  const int *string_id = game->string_id;
  const int other = FLIP_COLOR(color);
  int neighbor4[4];

  GetNeighbor4(neighbor4, pos);

  // 
  // 1
  // 2
  for (int i = 0; i < 4; i++) {
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


/////////////////////
//    //
/////////////////////
static void
MakeString( search_game_info_t *game, const int pos, const int color )
{
  string_t *string = game->string;
  string_t *new_string;
  char *board = game->board;
  int *string_id = game->string_id;
  int id = 1;
  int lib_add = 0;
  int other = FLIP_COLOR(color);
  int neighbor, neighbor4[4];

  // 
  while (string[id].flag) { id++; }

  // 
  new_string = &game->string[id];

  // 
  fill_n(new_string->lib, STRING_LIB_MAX, 0);
  fill_n(new_string->neighbor, MAX_NEIGHBOR, 0);
  new_string->color = (char)color;
  new_string->lib[0] = LIBERTY_END;
  new_string->neighbor[0] = NEIGHBOR_END;
  new_string->libs = 0;
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
  for (int i = 0; i < 4; i++) {
    if (board[neighbor4[i]] == S_EMPTY) {
      lib_add = AddLiberty(new_string, neighbor4[i], lib_add);
    } else if (board[neighbor4[i]] == other) {
      neighbor = string_id[neighbor4[i]];
      AddNeighbor(&string[neighbor], id);
      AddNeighbor(&string[id], neighbor);
    }
  }

  // 
  new_string->flag = true;
}


////////////////////
//    //
////////////////////
static void
MergeLiberty( string_t *dst, string_t *src )
{
  int dst_lib = 0, src_lib = 0;

  while (src_lib != LIBERTY_END) {
    // 
    if (dst->lib[src_lib] == 0) {
      // 
      while (dst->lib[dst_lib] < src_lib) {
	dst_lib = dst->lib[dst_lib];
      }
      // 
      dst->lib[src_lib] = dst->lib[dst_lib];
      dst->lib[dst_lib] = src_lib;
      // 1
      dst->libs++;
    }
    src_lib = src->lib[src_lib];
  }
}


////////////////
//    //
////////////////
static void
MergeStones( search_game_info_t *game, const int id, const int rm_id )
{
  string_t *string = game->string;
  int *string_next = game->string_next;
  int *string_id = game->string_id;
  int dst_pos = string[id].origin, src_pos = string[rm_id].origin;
  int pos;

  // srcdst
  // 
  if (dst_pos > src_pos) {
    // src
    pos = string_next[src_pos];
    // 
    string_next[src_pos] = dst_pos;
    // ID
    string_id[src_pos] = id;
    // 
    string[id].origin = src_pos;
    // 
    dst_pos = string[id].origin;
    // 1
    src_pos = pos;
  }

  while (src_pos != STRING_END) {
    string_id[src_pos] = id;
    pos = string_next[src_pos];
    // 
    while (string_next[dst_pos] < src_pos) {
      dst_pos = string_next[dst_pos];
    }
    // 
    string_next[src_pos] = string_next[dst_pos];
    string_next[dst_pos] = src_pos;
    // 
    src_pos = pos;
  }

  string[id].size += string[rm_id].size;
}


///////////////////
//    //
///////////////////
static void
MergeString( search_game_info_t *game, string_t *dst, string_t *src[3], const int n )
{
  int *string_id = game->string_id;
  int id = string_id[dst->origin], rm_id;
  string_t *string = game->string;

  for (int i = 0; i < n; i++) {
    // ID
    rm_id = string_id[src[i]->origin];
    // 
    MergeLiberty(dst, src[i]);
    // ID
    MergeStones(game, id, rm_id);
    // 
    MergeNeighbor(string, dst, src[i], id, rm_id);
    // 
    src[i]->flag = false;
  }
}


//////////////////////
//  1  //
//////////////////////
static void
RemoveLiberty( search_game_info_t *game, string_t *string, const int pos )
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


/////////////////////////////
//  ID  //
/////////////////////////////
static void
MergeNeighbor( string_t *string, string_t *dst, string_t *src, const int id, const int rm_id )
{
  int src_neighbor = 0, dst_neighbor = 0;
  int neighbor = src->neighbor[0];

  while (src_neighbor != NEIGHBOR_END) {
    // ID
    if (dst->neighbor[src_neighbor] == 0) {
      // 
      while (dst->neighbor[dst_neighbor] < src_neighbor) {
	dst_neighbor = dst->neighbor[dst_neighbor];
      }
      // ID
      dst->neighbor[src_neighbor] = dst->neighbor[dst_neighbor];
      dst->neighbor[dst_neighbor] = src_neighbor;
      // 1
      dst->neighbors++;
    }
    src_neighbor = src->neighbor[src_neighbor];
  }

  // src
  // rm_id
  while (neighbor != NEIGHBOR_END) {
    RemoveNeighborString(&string[neighbor], rm_id);
    AddNeighbor(&string[neighbor], id);
    neighbor = src->neighbor[neighbor];
  }
}


///////////////////////////
//  ID  //
///////////////////////////
static void
RemoveNeighborString( string_t *string, const int id )
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


///////////////////////
//    //
///////////////////////
static int
RemoveString( search_game_info_t *game, string_t *string )
{
  string_t *str = game->string;
  int *string_next = game->string_next;
  int *string_id = game->string_id;
  int pos = string->origin, next;
  char *board = game->board;
  bool *candidates = game->candidates;
  int neighbor, rm_id = string_id[string->origin];

  do {
    // 
    board[pos] = S_EMPTY;

    // 
    candidates[pos] = true;

    // 
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


////////////////
//    //
////////////////
static void
RestoreChain( search_game_info_t *game, const int id, const int stone[], const int stones, const int color )
{
  string_t *string = game->string;
  string_t *new_string;
  char *board = game->board;
  int *string_id = game->string_id;
  int lib_add = 0;
  const int other = FLIP_COLOR(color);
  int neighbor, neighbor4[4];
  int pos;

  // 
  new_string = &game->string[id];

  // 
  fill_n(new_string->lib, STRING_LIB_MAX, 0);
  fill_n(new_string->neighbor, MAX_NEIGHBOR, 0);
  new_string->color = (char)color;
  new_string->lib[0] = LIBERTY_END;
  new_string->neighbor[0] = NEIGHBOR_END;
  new_string->libs = 0;
  new_string->origin = stone[0];
  new_string->size = stones;
  new_string->neighbors = 0;

  for (int i = 0; i < stones; i++) {
    pos = stone[i];
    board[pos] = (char)color;
    game->string_id[pos] = id;
    UpdateMD2Stone(game->pat, color, pos); 
  }

  for (int i = 0; i < stones - 1; i++) {
    game->string_next[stone[i]] = stone[i + 1];
  }
  game->string_next[stone[stones - 1]] = STRING_END;

  for (int i = 0; i < stones; i++) {
    pos = stone[i];
    // 
    GetNeighbor4(neighbor4, pos);
    // 
    // , 
    // , 
    for (int j = 0; j < 4; j++) {
      if (board[neighbor4[j]] == S_EMPTY) {
	AddLiberty(new_string, neighbor4[j], lib_add);
      } else if (board[neighbor4[j]] == other) {
	neighbor = string_id[neighbor4[j]];
	RemoveLiberty(game, &string[neighbor], pos);	
	AddNeighbor(&string[neighbor], id);
	AddNeighbor(&string[id], neighbor);	
      }
    }
  }

  // 
  new_string->flag = true;
}


////////////////
//    //
////////////////
void
Undo( search_game_info_t *game )
{
  const int pm_count = game->moves - 1;
  const int previous_move = game->record[pm_count].pos;
  const int played_color = game->record[pm_count].color;
  const int opponent_color = FLIP_COLOR(played_color);
  string_t *string = game->string;
  int *string_id = game->string_id;
  undo_record_t* rec = &game->undo[pm_count];

  // 
  RemoveString(game, &string[string_id[previous_move]]);

  // 1
  for (int i = 0; i < rec->strings; i++) {
    if (rec->string_color[i] == opponent_color) {
      game->prisoner[played_color] -= rec->stones[i];
    }
    RestoreChain(game, rec->strings_id[i], rec->stone[i], rec->stones[i], rec->string_color[i]);
    rec->stones[i] = 0;
  }

  rec->strings = 0;

  game->ko_move = rec->ko_move_record;
  game->ko_pos = rec->ko_pos_record;

  game->moves--;
}

