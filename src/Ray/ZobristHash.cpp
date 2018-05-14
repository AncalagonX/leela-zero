#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "Nakade.h"
#include "ZobristHash.h"

using namespace std;


////////////
//    //
////////////

//  UCT ()
unsigned long long move_bit[MAX_RECORDS][BOARD_MAX][HASH_KO + 1]; 

// 
unsigned long long hash_bit[BOARD_MAX][HASH_KO + 1];

// 
unsigned long long shape_bit[BOARD_MAX];  

// 
node_hash_t *node_hash;

// 
static unsigned int used;

// 
static int oldest_move;

// 
unsigned int uct_hash_size = UCT_HASH_SIZE;

// 
unsigned int uct_hash_limit = UCT_HASH_SIZE * 9 / 10;

// 
bool enough_size;


////////////////////////////////////
//    //
////////////////////////////////////
void
SetHashSize( const unsigned int new_size )
{
  if (!(new_size & (new_size - 1))) {
    uct_hash_size = new_size;
    uct_hash_limit = new_size * 9 / 10;
  } else {
    cerr << "Hash size must be 2 ^ n" << endl;
    for (int i = 1; i <= 20; i++) {
      cerr << "2^" << i << ":" << (1 << i) << endl;
    }
    exit(1);
  }

}


/////////////////////////
//    //
/////////////////////////
unsigned int
TransHash( const unsigned long long hash )
{
  return ((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (uct_hash_size - 1);
}


/////////////////////
//  bit  //
/////////////////////
void
InitializeHash( void )
{
  std::random_device rnd;
  std::mt19937_64 mt(rnd());

  for (int i = 0; i < MAX_RECORDS; i++) {
    for (int j = 0; j < BOARD_MAX; j++) {
      move_bit[i][j][HASH_PASS] = mt();
      move_bit[i][j][HASH_BLACK] = mt();
      move_bit[i][j][HASH_WHITE] = mt();
      move_bit[i][j][HASH_KO] = mt();
    }
  }
    
  for (int i = 0; i < BOARD_MAX; i++) {  
    hash_bit[i][HASH_PASS]  = mt();
    hash_bit[i][HASH_BLACK] = mt();
    hash_bit[i][HASH_WHITE] = mt();
    hash_bit[i][HASH_KO]    = mt();
    shape_bit[i] = mt();
  }

  node_hash = new node_hash_t[uct_hash_size];

  if (node_hash == NULL) {
    cerr << "Cannot allocate memory" << endl;
    exit(1);
  }

  enough_size = true;

  InitializeNakadeHash();
}


//////////////////////////////////
//  UCT  //
//////////////////////////////////
void
InitializeUctHash( void )
{
  oldest_move = 1;
  used = 0;

  for (unsigned int i = 0; i < uct_hash_size; i++) {
    node_hash[i].flag = false;
    node_hash[i].hash = 0;
    node_hash[i].color = 0;
  }
}


/////////////////////////////////////
//  UCT  //
/////////////////////////////////////
void
ClearUctHash( void )
{
  used = 0;
  enough_size = true;

  for (unsigned int i = 0; i < uct_hash_size; i++) {
    node_hash[i].flag = false;
    node_hash[i].hash = 0;
    node_hash[i].color = 0;
    node_hash[i].moves = 0;
  }
}


////////////////////////////////////////
//    //
////////////////////////////////////////
void
ClearNotDescendentNodes( vector<int> &indexes )
{
  auto iter = indexes.begin();

  for (int i = 0; i < (int)uct_hash_size; i++) {
    if (*iter == i) {
      iter++;
    } else if (node_hash[i].flag) {
      node_hash[i].flag = false;
      node_hash[i].hash = 0;
      node_hash[i].color = 0;
      node_hash[i].moves = 0;
      used--;
    }
  }

  enough_size = true;
}


///////////////////////
//    //
///////////////////////
void
DeleteOldHash( const game_info_t *game )
{
  while (oldest_move < game->moves) {
    for (unsigned int i = 0; i < uct_hash_size; i++) {
      if (node_hash[i].flag && node_hash[i].moves == oldest_move) {
	node_hash[i].flag = false;
	node_hash[i].hash = 0;
	node_hash[i].color = 0;
	node_hash[i].moves = 0;
	used--;
      }
    }
    oldest_move++;
  }

  enough_size = true;
}


//////////////////////////////////////
//    //
//////////////////////////////////////
unsigned int
SearchEmptyIndex( const unsigned long long hash, const int color, const int moves )
{
  const unsigned int key = TransHash(hash);
  unsigned int i = key;

  do {
    if (!node_hash[i].flag) {
      node_hash[i].flag = true;
      node_hash[i].hash = hash;
      node_hash[i].moves = moves;
      node_hash[i].color = color;
      used++;
      if (used > uct_hash_limit) enough_size = false;
      return i;
    }
    i++;
    if (i >= uct_hash_size) i = 0;
  } while (i != key);

  return uct_hash_size;
}


////////////////////////////////////////////
//    //
////////////////////////////////////////////
unsigned int
FindSameHashIndex( const unsigned long long hash, const int color, const int moves)
{
  const unsigned int key = TransHash(hash);
  unsigned int i = key;

  do {
    if (!node_hash[i].flag) {
      return uct_hash_size;
    } else if (node_hash[i].hash == hash &&
	       node_hash[i].color == color &&
	       node_hash[i].moves == moves) {
      return i;
    }
    i++;
    if (i >= uct_hash_size) i = 0;
  } while (i != key);

  return uct_hash_size;
}


////////////////////////////////////////////////
//    //
////////////////////////////////////////////////
bool
CheckRemainingHashSize( void )
{
  return enough_size;
}

