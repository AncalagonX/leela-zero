#ifndef _COMMAND_H_
#define _COMMAND_H_

////////////
//    //
////////////
enum COMMAND {
  COMMAND_PLAYOUT,
  COMMAND_TIME,
  COMMAND_SIZE,
  COMMAND_CONST_TIME,
  COMMAND_THREAD,
  COMMAND_KOMI,
  COMMAND_HANDICAP,
  COMMAND_REUSE_SUBTREE,
  COMMAND_PONDERING,
  COMMAND_TREE_SIZE,
  COMMAND_NO_DEBUG,
  COMMAND_SUPERKO,
  COMMAND_MAX,
};


////////////
//    //
////////////

// 
void AnalyzeCommand( int argc, char **argv );

#endif
