#include <cstring>
#include <random>

#include "GoBoard.h"
#include "Message.h"
#include "Point.h"
#include "Rating.h"
#include "Simulation.h"

using namespace std;


////////////////////////////////
//    //
////////////////////////////////
void
Simulation( game_info_t *game, int starting_color, std::mt19937_64 *mt )
{
  int color = starting_color;
  int pos = -1;
  int length;
  int pass_count;

  // 
  length = MAX_MOVES - game->moves;
  if (length < 0) {
    return;
  }

  //   
  fill_n(game->sum_rate, 2, 0);
  fill(game->sum_rate_row[0], game->sum_rate_row[2], 0);
  fill(game->rate[0], game->rate[2], 0);

  pass_count = (game->record[game->moves - 1].pos == PASS && game->moves > 1);

  // 
  Rating(game, S_BLACK, &game->sum_rate[0], game->sum_rate_row[0], game->rate[0]);
  // 
  Rating(game, S_WHITE, &game->sum_rate[1], game->sum_rate_row[1], game->rate[1]);

  // 
  while (length-- && pass_count < 2) {
    // 
    pos = RatingMove(game, color, mt);
    // 
    PoPutStone(game, pos, color);
    // 
    pass_count = (pos == PASS) ? (pass_count + 1) : 0;
    // 
    color = FLIP_COLOR(color);
  }

}
