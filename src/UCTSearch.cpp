/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"
#include "UCTSearch.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "ThreadPool.h"
#include "TimeControl.h"
#include "Timing.h"
#include "Training.h"
#include "Utils.h"

using namespace Utils;

// TODO: REWRITE ALL MY ADDITIONS TO BE CLEANER, AND SEPARATE INTO INDIVIDUAL FEATURE COMMITS

int last_update2 = 0;
int last_update3 = 0;
int last_update4 = 0;
int playouts2 = 0;
int playouts_stop = 0;
int playouts3 = 0;
int playouts_stop3 = 0;
int playout_rate_average = 1000;
int my_accumulated_improved_playout_rate = 1;
int average_counter = 0;
int min_required_visits2 = 0;
int added_time = 0;
bool pondering_now = false;

UCTSearch::UCTSearch(GameState& g)
    : m_rootstate(g) {
    set_playout_limit(cfg_max_playouts);
    set_visit_limit(cfg_max_visits);
    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
}

bool UCTSearch::advance_to_new_rootstate() {
    if (!m_root || !m_last_rootstate) {
        // No current state
        return false;
    }

    if (m_rootstate.get_komi() != m_last_rootstate->get_komi()) {
        return false;
    }

    auto depth =
        (int) (m_rootstate.get_movenum() - m_last_rootstate->get_movenum());

    if (depth < 0) {
        return false;
    }

    auto test = std::make_unique<GameState>(m_rootstate);
    for (auto i = 0; i < depth; i++) {
        test->undo_move();
    }

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // m_rootstate and m_last_rootstate don't match
        return false;
    }

    // Try to replay moves advancing m_root
    for (auto i = 0; i < depth; i++) {
        test->forward_move();
        const auto move = test->get_last_move();
        m_root = m_root->find_child(move);
        if (!m_root) {
            // Tree hasn't been expanded this far
            return false;
        }
        m_last_rootstate->play_move(move);
    }

    assert(m_rootstate.get_movenum() == m_last_rootstate->get_movenum());

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // Can happen if user plays multiple moves in a row by same player
        return false;
    }

    return true;
}

void UCTSearch::update_root() {
    // Definition of m_playouts is playouts per search call.
    // So reset this count now.
    m_playouts = 0;
	last_update2 = 0;
	last_update3 = 0;
	playouts2 = 0;
	playouts_stop = 0;
	int playouts3 = 0;
	int playouts_stop3 = 0;
	added_time = 0;
	cfg_fpu_reduction = 0.25;
	cfg_puct = 0.8;

//#ifndef NDEBUG
    auto start_nodes = m_root->count_nodes();
//#endif

    if (!advance_to_new_rootstate() || !m_root) {
        m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    }
    // Clear last_rootstate to prevent accidental use.
    m_last_rootstate.reset(nullptr);

    // Check how big our search tree (reused or new) is.
    m_nodes = m_root->count_nodes();

//#ifndef NDEBUG
    if (m_nodes > 0) {
        myprintf("updated root, %d -> %d nodes (%.1f%% reused)\n",
            start_nodes, m_nodes.load(), 100.0 * m_nodes.load() / start_nodes);
    }
//#endif
}

SearchResult UCTSearch::play_simulation(GameState & currstate,
                                        UCTNode* const node) {
    const auto color = currstate.get_to_move();
	const auto movenum = m_rootstate.get_movenum();
    auto result = SearchResult{};

    node->virtual_loss();

	if (!node->has_children()) {
		if (currstate.get_passes() >= 2) {
			auto score = currstate.final_score();
			result = SearchResult::from_score(score);
		}
		// TODO: IF "ROOT NODE" IS TRUE (code here) ----> node == m_root.get() <---- AND NOT "NODE->HAS CHILDREN" THEN GET AN EVAL.
		// TODO: AND THEN REMOVE THE UCTNODE CODE THAT GETS 1 VISIT PER ROOT NODE, AND SEE IF IT STILL GETS 1 PER EACH!
		// TODO: ALSO CHANGE TABLE OUTPUT TO 500 500 AGAIN.
		else if (m_nodes < MAX_TREE_SIZE) {
			float eval;
			auto success = node->create_children(m_nodes, currstate, eval);
			if (success) {
				result = SearchResult::from_eval(eval);
			}
		}
	}

	//if (node == m_root.get() && node->get_visits() < 10 && !result.valid()) {
	//		auto next = node->uct_select_child(color, node == m_root.get(), movenum, pondering_now);

	//	if (next != nullptr) {
	//		auto move = next->get_move();

	//		currstate.play_move(move);
	//		if (move != FastBoard::PASS && currstate.superko()) {
	//			next->invalidate();
	//		}
	//		else {
	//			result = play_simulation(currstate, next);
	//		}
	//	}
	//}

    if (node->has_children() && !result.valid()) {
		auto next = node->uct_select_child(color, node == m_root.get(), movenum, pondering_now, static_cast<int>(m_playouts));

        if (next != nullptr) {
            auto move = next->get_move();

            currstate.play_move(move);
            if (move != FastBoard::PASS && currstate.superko()) {
                next->invalidate();
            } else {
                result = play_simulation(currstate, next);
            }
        }
    }

    if (result.valid()) {
        node->update(result.eval());
    }
    node->virtual_loss_undo();

    return result;
}

void UCTSearch::dump_stats(FastState & state, UCTNode & parent) {
	if (cfg_quiet || !parent.has_children()) {
		return;
	}

	const int color = state.get_to_move();

	// sort children, put best move on top
	//parent.sort_children_reverse(color);
	parent.sort_children(color);

	if (parent.get_first_child()->first_visit()) {
		return;
	}

	int movecount = 0;
	for (const auto& node : parent.get_children()) {
		// Always display at least two moves. In the case there is
		// only one move searched the user could get an idea why.
		if (++movecount > 2 && !node->get_visits()) break;

		std::string move = state.move_to_text(node->get_move());
		FastState tmpstate = state;
		tmpstate.play_move(node->get_move());
		std::string pv = move + " " + get_pv(tmpstate, *node);

		//     myprintf("%4s -> %7d (V: %5.2f%%) (N: %5.2f%%) (UCB: %5.2f%%) (Old Value: %5.2f%%) PV: %s\n",
		//         move.c_str(),
		//         node->get_visits(),
		//node->get_lcb(color) * 100.0f, // I moved this here
		//         //node->get_visits() ? node->get_eval(color)*100.0f : 0.0f,
		//         node->get_score() * 100.0f,
		//         //node->get_lcb(color) * 100.0f,
		//node->get_ucb(color) * 100.0f,
		//node->get_visits() ? node->get_eval(color)*100.0f : 0.0f, // I moved this here
		//         pv.c_str());
		myprintf("%4s -> %7d (V: %5.2f%%) (N: %5.2f%%) PV: %s\n",
			move.c_str(),
			node->get_visits(),
			node->get_visits() ? node->get_eval(color)*100.0f : 0.0f,
			node->get_score() * 100.0f,
			pv.c_str());
	}
	tree_stats(parent);
}

void tree_stats_helper(const UCTNode& node, size_t depth,
                       size_t& nodes, size_t& non_leaf_nodes,
                       size_t& depth_sum, size_t& max_depth,
                       size_t& children_count) {
    nodes += 1;
    non_leaf_nodes += node.get_visits() > 1;
    depth_sum += depth;
    if (depth > max_depth) max_depth = depth;

    for (const auto& child : node.get_children()) {
        if (!child->first_visit()) children_count += 1;

        tree_stats_helper(*(child.get()), depth+1,
                          nodes, non_leaf_nodes, depth_sum,
                          max_depth, children_count);
    }
}

void UCTSearch::tree_stats(const UCTNode& node) {
    size_t nodes = 0;
    size_t non_leaf_nodes = 0;
    size_t depth_sum = 0;
    size_t max_depth = 0;
    size_t children_count = 0;
    tree_stats_helper(node, 0,
                      nodes, non_leaf_nodes, depth_sum,
                      max_depth, children_count);

    if (nodes > 0) {
        myprintf("%.1f average depth, %d max depth\n",
                 (1.0f*depth_sum) / nodes, max_depth);
        //myprintf("%d non leaf nodes, %.2f average children\n", non_leaf_nodes, (1.0f*children_count) / non_leaf_nodes);
    }
}

bool UCTSearch::should_resign(passflag_t passflag, float bestscore) {
    if (passflag & UCTSearch::NORESIGN) {
        // resign not allowed
        return false;
    }

    if (cfg_resignpct == 0) {
        // resign not allowed
        return false;
    }

    const auto visits = m_root->get_visits();
    if (visits < std::min(500, cfg_max_playouts))  {
        // low visits
        return false;
    }

    const size_t board_squares = m_rootstate.board.get_boardsize()
                               * m_rootstate.board.get_boardsize();
    const auto move_threshold = board_squares / 4;
    const auto movenum = m_rootstate.get_movenum();
    if (movenum <= move_threshold) {
        // too early in game to resign
        return false;
    }

    const auto color = m_rootstate.board.get_to_move();

    const auto is_default_cfg_resign = cfg_resignpct < 0;
    const auto resign_threshold =
        0.01f * (is_default_cfg_resign ? 10 : cfg_resignpct);
    if (bestscore > resign_threshold) {
        // eval > cfg_resign
        return false;
    }

    if ((m_rootstate.get_handicap() > 0)
            && (color == FastBoard::WHITE)
            && is_default_cfg_resign) {
        const auto handicap_resign_threshold =
            resign_threshold / (1 + m_rootstate.get_handicap());

        // Blend the thresholds for the first ~215 moves.
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * board_squares));
        auto blended_resign_threshold = blend_ratio * resign_threshold
            + (1 - blend_ratio) * handicap_resign_threshold;
        if (bestscore > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    return true;
}

int UCTSearch::get_best_move(passflag_t passflag) {
    int color = m_rootstate.board.get_to_move();

    // Make sure best is first
    m_root->sort_children(color);

    // Check whether to randomize the best move proportional
    // to the playout counts, early game only.
    auto movenum = int(m_rootstate.get_movenum());
    if (movenum < cfg_random_cnt) {
        m_root->randomize_first_proportionally();
    }

    auto first_child = m_root->get_first_child();
    assert(first_child != nullptr);

    auto bestmove = first_child->get_move();
    auto bestscore = first_child->get_eval(color);

    // do we want to fiddle with the best move because of the rule set?
    if (passflag & UCTSearch::NOPASS) {
        // were we going to pass?
        if (bestmove == FastBoard::PASS) {
            UCTNode * nopass = m_root->get_nopass_child(m_rootstate);

            if (nopass != nullptr) {
                myprintf("Preferring not to pass.\n");
                bestmove = nopass->get_move();
                if (nopass->first_visit()) {
                    bestscore = 1.0f;
                } else {
                    bestscore = nopass->get_eval(color);
                }
            } else {
                myprintf("Pass is the only acceptable move.\n");
            }
        }
    } else {
        if (!cfg_dumbpass && bestmove == FastBoard::PASS) {
            // Either by forcing or coincidence passing is
            // on top...check whether passing loses instantly
            // do full count including dead stones.
            // In a reinforcement learning setup, it is possible for the
            // network to learn that, after passing in the tree, the two last
            // positions are identical, and this means the position is only won
            // if there are no dead stones in our own territory (because we use
            // Trump-Taylor scoring there). So strictly speaking, the next
            // heuristic isn't required for a pure RL network, and we have
            // a commandline option to disable the behavior during learning.
            // On the other hand, with a supervised learning setup, we fully
            // expect that the engine will pass out anything that looks like
            // a finished game even with dead stones on the board (because the
            // training games were using scoring with dead stone removal).
            // So in order to play games with a SL network, we need this
            // heuristic so the engine can "clean up" the board. It will still
            // only clean up the bare necessity to win. For full dead stone
            // removal, kgs-genmove_cleanup and the NOPASS mode must be used.
            float score = m_rootstate.final_score();
            // Do we lose by passing?
            if ((score > 0.0f && color == FastBoard::WHITE)
                ||
                (score < 0.0f && color == FastBoard::BLACK)) {
                myprintf("Passing loses :-(\n");
                // Find a valid non-pass move.
                UCTNode * nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr) {
                    myprintf("Avoiding pass because it loses.\n");
                    bestmove = nopass->get_move();
                    if (nopass->first_visit()) {
                        bestscore = 1.0f;
                    } else {
                        bestscore = nopass->get_eval(color);
                    }
                } else {
                    myprintf("No alternative to passing.\n");
                }
            } else {
                myprintf("Passing wins :-)\n");
            }
        } else if (!cfg_dumbpass
                   && m_rootstate.get_last_move() == FastBoard::PASS) {
            // Opponents last move was passing.
            // We didn't consider passing. Should we have and
            // end the game immediately?
            float score = m_rootstate.final_score();
            // do we lose by passing?
            if ((score > 0.0f && color == FastBoard::WHITE)
                ||
                (score < 0.0f && color == FastBoard::BLACK)) {
                myprintf("Passing loses, I'll play on.\n");
            } else {
                myprintf("Passing wins, I'll pass out.\n");
                bestmove = FastBoard::PASS;
            }
        }
    }

    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
        if (should_resign(passflag, bestscore)) {
            myprintf("Eval (%.2f%%) looks bad. Resigning.\n",
                     100.0f * bestscore);
            bestmove = FastBoard::RESIGN;
        }
    }
	myprintf("Move %d\n", movenum);
    return bestmove;
}

std::string UCTSearch::get_pv(FastState & state, UCTNode& parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.get_to_move());
    if (best_child.first_visit()) {
        return std::string();
    }
    auto best_move = best_child.get_move();
    auto res = state.move_to_text(best_move);

    state.play_move(best_move);

    auto next = get_pv(state, best_child);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    return res;
}

void UCTSearch::dump_analysis(int playouts) {
    if (cfg_quiet) {
        return;
    }

    FastState tempstate = m_rootstate;
    int color = tempstate.board.get_to_move();

    std::string pvstring = get_pv(tempstate, *m_root);
    float winrate = 100.0f * m_root->get_eval(color);
    myprintf("Playouts: %d, Win: %5.2f%%, PV: %s\n",
             playouts, winrate, pvstring.c_str());
}

bool UCTSearch::is_running() const {
    return m_run && m_nodes < MAX_TREE_SIZE;
}

int UCTSearch::est_playouts_left(int elapsed_centis, int time_for_move, int playouts) const
//{
//    auto playouts = m_playouts.load();
//    const auto playouts_left = std::min(m_maxplayouts - playouts,
//                                        m_maxvisits - m_root->get_visits());
//
//    // Wait for at least 1 second and 1000 playouts
//    // so we get a reliable playout_rate.
//	if (elapsed_centis < 300 && playouts < 3000) {
//        return playouts_left;
//    }
//    const auto playout_rate = 1.0f * playouts / elapsed_centis;
//	if (playout_rate > 10000) {
//		const auto playout_rate = 2000;
//	}
//    const auto time_left = time_for_move - elapsed_centis;
//    return std::min(playouts_left, static_cast<int>(std::ceil(playout_rate * time_left)));
//}


{
	auto playouts_calc = m_playouts.load();
	const auto playouts_left = std::min(m_maxplayouts - playouts_calc,
		m_maxvisits - m_root->get_visits());



	playouts_stop3 = playouts;

	// check every 1 second whether playout_rate has dropped under 10000 playouts/sec
	if (elapsed_centis > 50 && playouts_stop3 > 1000) {
		if (elapsed_centis - last_update3 >= 100) { // run once every 1 second
			playouts_stop3 = playouts;
			if (playouts3 >= playouts_stop3) {
				playouts3 = 0;
			}
			auto playout_rate = (playouts_stop3 - playouts3); // EVEN BETTER playout_rate calc
																  //myprintf("\n     %d playouts_stop\n", playouts_stop);
																  //myprintf("\n     %d elapsed_centis - last_update3\n", (elapsed_centis - last_update3));
																  //myprintf("\n     %d playouts2\n", playouts2);
			last_update3 = elapsed_centis;
			last_update3 = (last_update3 - (last_update3 % 100));
			if (playout_rate >= 10000 || playout_rate <= 100) { // if playout_rate is >= 10000 or <= 100, then set "est_playouts_left" function to (5 * m_maxplayouts) and recalc 1 second later
				playouts3 = playouts_stop3;
				const auto time_left2 = ((time_for_move + added_time) - elapsed_centis);
				//myprintf("\n%d playout_rate\n", static_cast<int>(playout_rate));
				//myprintf("\n%d time_for_move\n", static_cast<int>(time_for_move));
				//myprintf("\n%d elapsed_centis\n", static_cast<int>(elapsed_centis));
				//myprintf("\n%d time_left2\n", static_cast<int>(time_left2));
				//myprintf("\n%d playout_rate * time_left\n", static_cast<int>(std::ceil(playout_rate * time_left2)));
				const auto playout_rate_original_calc = 1.0f * playouts3 / elapsed_centis;
				//myprintf("\n%d playout_rate_original_calc * time_left\n", static_cast<int>(std::ceil(playout_rate_original_calc * time_left2)));
				const auto my_improved_playout_rate = 1.0f * playout_rate; // / (elapsed_centis - last_update3);
				//myprintf("\n%d my_improved_playout_rate * time_left / 100 \n", static_cast<int>(std::ceil((playout_rate * time_left2) / 100)));
				//myprintf("\nover 10k RETURNED %d est_playouts_left\n", static_cast<int>(5 * m_maxplayouts));
				return (5 * m_maxplayouts); // set this "est_playouts_left" function to return the max ponderable playouts
			}
			else if (playout_rate > 100 && playout_rate < 10000) {
				playouts3 = playouts_stop3;
				const auto time_left2 = ((time_for_move + added_time) - elapsed_centis);
				//myprintf("\n%d playout_rate\n", static_cast<int>(playout_rate));
				//myprintf("\n%d time_for_move\n", static_cast<int>(time_for_move));
				//myprintf("\n%d elapsed_centis\n", static_cast<int>(elapsed_centis));
				//myprintf("\n%d time_left2\n", static_cast<int>(time_left2));
				//myprintf("\n%d playout_rate * time_left\n", static_cast<int>(std::ceil(playout_rate * time_left2)));
				//const auto playout_rate_original_calc = 1.0f * playouts3 / elapsed_centis;
				//myprintf("\n%d playout_rate_original_calc * time_left\n", static_cast<int>(std::ceil(playout_rate_original_calc * time_left2)));
				//const auto my_improved_playout_rate = 1.0f * playout_rate; // / (elapsed_centis - last_update3);
				//auto my_improved_playout_rate = static_cast<int>(std::ceil((playout_rate * time_left2) / 100));
				//myprintf("\n%d my_improved_playout_rate\n", my_improved_playout_rate);

				average_counter = average_counter + 1;

				//myprintf("\n%d average_counter\n", average_counter);
				//if ((my_accumulated_improved_playout_rate / average_counter) <= (2 * playout_rate)) {
				//	my_accumulated_improved_playout_rate = (my_accumulated_improved_playout_rate + (0.75 * playout_rate));
				//	myprintf("\nStatement 1\n");
				//}
				//else if ((my_accumulated_improved_playout_rate / average_counter) >= (2 * playout_rate)) {
				//	my_accumulated_improved_playout_rate = (my_accumulated_improved_playout_rate + (1.33 * playout_rate));
				//	myprintf("\nStatement 2\n");
				//}
				//else {
				//	my_accumulated_improved_playout_rate = (my_accumulated_improved_playout_rate + playout_rate);
				//	myprintf("\nStatement 3\n");
				//}
				if (playout_rate >= 4 * playout_rate_average) {
					my_accumulated_improved_playout_rate = (my_accumulated_improved_playout_rate + (playout_rate / ((playout_rate / playout_rate_average) - 2)));
				}
				else {
					my_accumulated_improved_playout_rate = (my_accumulated_improved_playout_rate + playout_rate);
				}
				//myprintf("\n%d my_accumulated_improved_playout_rate\n", my_accumulated_improved_playout_rate);
				playout_rate_average = ((my_accumulated_improved_playout_rate) / average_counter);
				auto est_playouts_left_value = std::min(playouts_left, static_cast<int>(std::ceil((playout_rate_average * time_left2) / 100)));
				auto est_time_remaining_now = ((1.0f * est_playouts_left_value) / (1.0f * playout_rate_average));
				myprintf("\n%d/s average -> (%.1fs estimated time left)\n", playout_rate_average, est_time_remaining_now);
				if (average_counter >= 60) {
					average_counter = (average_counter - (0.25 * average_counter));
					my_accumulated_improved_playout_rate = (my_accumulated_improved_playout_rate - (0.25 * my_accumulated_improved_playout_rate));
				}
				//myprintf("\n%d playout_rate_average * time_left / 100 \n", static_cast<int>(std::ceil((playout_rate_average * time_left2) / 100)));
				//myprintf("\nunder 10k RETURNED %d est_playouts_left\n", std::min(playouts_left, static_cast<int>(std::ceil((playout_rate_average * time_left2) / 100))));
				//if (playout_rate_average % 10) {
				//	my_accumulated_improved_playout_rate = static_cast<int>(my_accumulated_improved_playout_rate * 0.9);
				//	playout_rate_average = static_cast<int>(playout_rate_average * 0.9);
				//}
				//return std::min(playouts_left, static_cast<int>(std::ceil((playout_rate_average * time_left2) / 100)));
				return est_playouts_left_value;
			}
		}
	}
	else {
		return (5 * m_maxplayouts);
	}
}









size_t UCTSearch::prune_noncontenders(int elapsed_centis, int time_for_move, int original_time_for_move) {
	auto Nfirst = 1;
	auto Nfirst_eval = 0.01f;
	auto Nfirst_score = 0.01f;
	//auto Nfirst_eval_old = 0.01f;
	//auto Nfirst_score_old = 0.01f;
	auto Nsecond = 1;
	auto Nsecond_eval = 0.01f;
	auto Nsecond_score = 0.01f;
	//auto Nsecond_eval_old = 0.01f;
	//auto Nsecond_score_old = 0.01f;
	auto est_playouts_left2 = 0;
	auto est_playouts_left3 = 0;
	auto min_required_visits = 0;
	auto min_required_visits2 = 0;
	int color = m_rootstate.board.get_to_move();

	for (const auto& node : m_root->get_children()) {
		if (node->valid()) {
			Nfirst = std::max(Nfirst, node->get_visits());
			//Nfirst_eval = std::max((Nfirst_eval * 1.00f), (node->get_visits() ? node->get_eval(color)*100.0f : 0.01f));
			if (node->valid() && node->get_visits() == Nfirst) {
				Nfirst_eval = std::max((Nfirst_eval * 1.00f), (node->get_visits() ? node->get_eval(color)*100.0f : 0.01f));
				Nfirst_score = std::max((Nfirst_score * 1.00f), (node->get_visits() ? node->get_score()*100.0f : 0.01f));
			}
		}
	}
	////node->get_score() * 100.0f
	for (const auto& node : m_root->get_children()) {
		if (node->valid() && node->get_visits() < Nfirst) {
			Nsecond = std::max(Nsecond, node->get_visits());
			if (node->valid() && node->get_visits() == Nsecond) {
				Nsecond_eval = std::max((Nsecond_eval * 1.00f), (node->get_visits() ? node->get_eval(color)*100.0f : 0.01f));
				Nsecond_score = std::max((Nsecond_score * 1.00f), (node->get_visits() ? node->get_score()*100.0f : 0.01f));
			}
		}
    }
	est_playouts_left2 = est_playouts_left(elapsed_centis, time_for_move, static_cast<int>(m_playouts));
    min_required_visits = (((3 * Nfirst) + Nsecond) / 4) - est_playouts_left2;
	if (elapsed_centis > 100 && est_playouts_left3 != est_playouts_left2) {
		//myprintf("\n%d min_required_visits\n", static_cast<int>(min_required_visits));
		//myprintf("\n%d Nfirst\n", static_cast<int>(Nfirst));
		//myprintf("\n%d Nsecond\n", static_cast<int>(Nsecond));
		auto Nfirst_Nsecond_ratio = (1.0f * (1.0f * (Nfirst)) / (1.0f * (Nsecond)));
		auto Nfirst_eval_Nsecond_eval_ratio = (1.0f * (1.0f * (Nfirst_eval)) / (1.0f * (Nsecond_eval)));
		auto Nfirst_score_Nsecond_score_ratio = (1.0f * (1.0f * (Nfirst_score)) / (1.0f * (Nsecond_score)));
		myprintf("\nVISIT RATIO: %6.1f -> (%d / %d) -> Nfirst / Nsecond\n", (1.0f * Nfirst_Nsecond_ratio), Nfirst, Nsecond);
		myprintf("\nVALUE RATIO:   %6.2f -> (%5.2f%% / %5.2f%%) -> Nfirst_eval / Nsecond_eval\n", (1.0f * Nfirst_eval_Nsecond_eval_ratio), (1.0f * (Nfirst_eval)), (1.0f * (Nsecond_eval)));
		myprintf("\nPOLICY RATIO:  %6.2f -> (%5.2f%% / %5.2f%%) -> Nfirst_score / Nsecond_score\n", (1.0f * Nfirst_score_Nsecond_score_ratio), (1.0f * (Nfirst_score)), (1.0f * (Nsecond_score)));
		//myprintf("\n%d est_playouts_left\n", (1 - static_cast<int>(est_playouts_left2)));
		//myprintf("\n%d playout_rate_average\n", playout_rate_average);
		//myprintf("\n%.1fs left\n", (1.0f * (1.0f * (min_required_visits)) / (1.0f * (playout_rate_average))));
		//myprintf("\n%.1fs time_for_move\n", ((1.0f * (time_for_move)) / 100));
		//myprintf("\n%.1fs remaining\n", ((1.0f * (time_for_move - elapsed_centis)) / 100));

		//int color = m_rootstate.board.get_to_move();
		//auto bestscore = first_child->get_eval(color);
		
		const auto movenum = m_rootstate.get_movenum();
		
		//if (0.99 <= Nfirst_eval_Nsecond_eval_ratio && 1.01 >= Nfirst_eval_Nsecond_eval_ratio && static_cast<int>(m_playouts) > 5000) {
		

		//////////// UNCOMMENT THE BELOW BLOCK LATER IF I NEED QUICKER MOVES IN EARLY FUSEKI
		/*   if (0.99 <= Nfirst_eval_Nsecond_eval_ratio && 1.01 >= Nfirst_eval_Nsecond_eval_ratio && movenum <= 19 && static_cast<int>(m_playouts) > 5000) {
			added_time = (added_time - 300);
			//myprintf("\n%6.2f -> (%5.2f%% / %5.2f%%) -> Nfirst_eval / Nsecond_eval\n", (1.0f * Nfirst_eval_Nsecond_eval_ratio), (1.0f * (Nfirst_eval)), (1.0f * (Nsecond_eval)));			
		}

		*/

		if (Nfirst_eval < 40.0 && (added_time + time_for_move + 300) <= original_time_for_move) {
			added_time = (added_time + 300);
		}
		if (1.05 <= Nfirst_eval_Nsecond_eval_ratio && Nfirst_Nsecond_ratio > 100 && static_cast<int>(Nfirst) > 5000) {
			added_time = (added_time - 300);
		}
		if (Nfirst_Nsecond_ratio < 3 && (added_time + time_for_move + 150) <= original_time_for_move) {
			added_time = (added_time + 150);
			//myprintf("\n%.1fs + %.1fs now remaining\n", ((1.0f * (time_for_move - elapsed_centis)) / 100), ((1.0f * added_time) / 100));
			//myprintf("\n%.1fs total added_time\n", ((1.0f * added_time) / 100));
		}
		if (Nfirst_Nsecond_ratio > 5 && (added_time > 0) && (added_time + time_for_move) <= original_time_for_move) {
			added_time = (added_time - 75);
			//myprintf("\n%.1fs + %.1fs now remaining\n", ((1.0f * (time_for_move - elapsed_centis)) / 100), ((1.0f * added_time) / 100));
			//myprintf("\n%.1fs total added_time\n", ((1.0f * added_time) / 100));
		}
		
		//////////////////IF SCOREFIRST > SCORESECOND THEN DECREASE PUCT







		
		/* if (Nfirst_score_Nsecond_score_ratio)

		if (Nfirst_Nsecond_ratio >= 5) {
			cfg_puct = cfg_puct + 0.10;
			if (cfg_puct > 1.5) {
				cfg_puct = 1.5;
			}
		}
		else if (Nfirst_Nsecond_ratio <= 2) {
			cfg_puct = cfg_puct - 0.10;
			if (cfg_puct < 0.20) {
				cfg_puct = 0.20;
			}
		}
		else if ((2 < Nfirst_Nsecond_ratio && Nfirst_Nsecond_ratio < 5)) {
			cfg_puct = (0.8 + (3 * cfg_puct) / 4);
		}
		else {
			cfg_puct = 0.80;
		}
		if (Nfirst_eval_Nsecond_eval_ratio < 1 && static_cast<int>(m_playouts) > 5000) {
			added_time = (added_time + 150);
			cfg_puct = cfg_puct - 0.10;
			if (cfg_puct < 0.20) {
				cfg_puct = 0.20;
			}
		} */








		//if (cfg_fpu_reduction < 0 && static_cast<int>(m_playouts) > 1000) {
		//	cfg_fpu_reduction = (cfg_fpu_reduction / 2) + 1;
		//}

		//if (cfg_fpu_reduction < 0.50 && static_cast<int>(m_playouts) > 1000 && movenum < 20) {
		//	cfg_fpu_reduction = cfg_fpu_reduction + 0.05;
		//}
		//if (cfg_fpu_reduction > 0.50 && movenum < 20) {
		//	cfg_fpu_reduction = 0.50;
		//}

		//added_time = 0;

		//if (Nfirst_eval_Nsecond_eval_ratio < 1 && static_cast<int>(m_playouts) > 5000) {
		//	added_time = (added_time + 300);
		//	cfg_fpu_reduction = cfg_fpu_reduction + 0.10;
		//	if (cfg_fpu_reduction > 0.50) {
		//		cfg_fpu_reduction = 0.50;
		//	}
		//}
		//if (Nfirst_eval_Nsecond_eval_ratio >= 1.01 && static_cast<int>(m_playouts) > 5000) {
		//	cfg_fpu_reduction = 0.25;
		//	if (cfg_fpu_reduction < 0.05) {
		//		cfg_fpu_reduction = 0.05;
		//	}
		//}
		myprintf("\n%.1fs time pool remaining -> (%.1fs added) -> %4.1f/s max time per move\n", (((1.0f * (time_for_move - elapsed_centis)) / 100) + ((1.0f * added_time) / 100)), ((1.0f * added_time) / 100), ((1.0f * original_time_for_move) / 100));
		est_playouts_left3 = est_playouts_left2;
	}
    auto pruned_nodes = size_t{0};
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
             const auto has_enough_visits = node->get_visits() >= min_required_visits;
             node->set_active(has_enough_visits);
             if (!has_enough_visits) {
                 ++pruned_nodes;
             }
        }
    }

    assert(pruned_nodes < m_root->get_children().size());
    return pruned_nodes;
}

bool UCTSearch::stop_thinking(int elapsed_centis, int time_for_move, int playouts) const {
	//auto playouts = m_playouts.load();
	//const auto playouts_left = std::min(m_maxplayouts - playouts,
	//	m_maxvisits - m_root->get_visits());

	// Wait for at least 1 second and 1000 playouts
	// so we get a reliable playout_rate.
	//if (elapsed_centis < 100 || playouts < 1000) {
	//	return playouts_left;
	//}

	//Time start2;
	//int playout_rate_stop = playouts_stop - playouts2;
	playouts_stop = playouts;
	//Time elapsed2;
	//int elapsed_centis2 = Time::timediff_centis(start2, elapsed2);

	// check every 1 second whether playout_rate has dropped under 2000 playouts/sec
	if (elapsed_centis > 99 && playouts_stop > 2000) {
		if (elapsed_centis - last_update2 >= 100) { // run once every 1 second
			playouts_stop = playouts;
			// MOVED THIS UP //auto playouts = m_playouts.load(); // get up to date total playouts for this turn
			//const auto playout_rate = 1.0f * playouts / elapsed_centis; //calc playout_rate
			//const auto playout_rate_stop = ((playouts_stop - playouts2) * 100.0) / ((elapsed_centis - last_update2) + 1); // BETTER playout_rate calc
			auto playout_rate_stop = (playouts_stop - playouts2); // EVEN BETTER playout_rate calc
			//myprintf("\n%d playouts/sec for last 1 second\n", playout_rate_stop);
			//myprintf("\n     %d playouts_stop\n", playouts_stop);
			//myprintf("\n     %d elapsed_centis - last_update2\n", (elapsed_centis - last_update2));
			//myprintf("\n     %d playouts2\n", playouts2);
			last_update2 = elapsed_centis;
			if (playout_rate_stop >= 10000) {//if playout_rate more than 10000, then set playouts2 to playouts_stop and recalc 1 second later
				playouts2 = playouts_stop;
				return false;
			}
			else if (playout_rate_stop < 10000) {
				playouts2 = playouts_stop;
				return m_playouts >= m_maxplayouts
					|| m_root->get_visits() >= m_maxvisits
					|| elapsed_centis >= (time_for_move + added_time);
			}
		}
	}
	else {
		return false;
	}

//	const auto playout_rate2 = 0;
//	const auto playout_rate = 1.0f * playouts / elapsed_centis;
//	if (playout_rate > 2000) {
//		const auto playout_rate = 2000;
//	}
//	return m_playouts >= m_maxplayouts
//           || m_root->get_visits() >= m_maxvisits
//           || elapsed_centis >= time_for_move;
}

bool UCTSearch::stop_thinking_pondering(int elapsed_centis, int time_for_move, int playouts) const {
	return m_playouts >= 5 * m_maxplayouts
		|| m_root->get_visits() >= 5 * m_maxvisits;
}


void UCTWorker::operator()() {
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = m_search->play_simulation(*currstate, m_root);
        if (result.valid()) {
            m_search->increment_playouts();
        }
    } while(m_search->is_running());
}

void UCTSearch::increment_playouts() {
    m_playouts++;
}

int UCTSearch::think(int color, passflag_t passflag) {
    // Start counting time for us
    m_rootstate.start_clock(color);

    update_root();
    // set side to move
    m_rootstate.board.set_to_move(color);
	const auto movenum = m_rootstate.get_movenum();
	//if (movenum <= 19) {
	//	cfg_fpu_reduction = 0.25;
	//}
	//else {
	//	cfg_fpu_reduction = 0.25;
	//}
    // set up timing info
    Time start;

    m_rootstate.get_timecontrol().set_boardsize(m_rootstate.board.get_boardsize());
	auto original_time_for_move = m_rootstate.get_timecontrol().max_time_for_move(color);
    //auto time_for_move = m_rootstate.get_timecontrol().max_time_for_move(color);
	
	auto time_for_move = original_time_for_move;
	if (cfg_timemanage == TimeManagement::ON) {
		time_for_move = (original_time_for_move / 2);
	}
	else {
		time_for_move = original_time_for_move;
	}

    myprintf("Thinking at most %.1f seconds...\n", time_for_move/100.0f);

    // create a sorted list off legal moves (make sure we
    // play something legal and decent even in time trouble)
    float root_eval;
    if (!m_root->has_children()) {
        m_root->create_children(m_nodes, m_rootstate, root_eval);
        m_root->update(root_eval);
    } else {
        root_eval = m_root->get_eval(color);
    }
    m_root->kill_superkos(m_rootstate);
    if (cfg_noise) {
        // Adjusting the Dirichlet noise's alpha constant to the board size
        auto alpha = 0.03f * 361.0f / BOARD_SQUARES;

        m_root->dirichlet_noise(0.25f, alpha);
    }

    myprintf("NN eval=%f\n",
             (color == FastBoard::BLACK ? root_eval : 1.0f - root_eval));

    m_run = true;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }

    bool keeprunning = true;
    int last_update = 0;
	int last_update4 = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }

        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);

        // output some stats every few seconds
        // check if we should still search
        if (elapsed_centis - last_update > 99) {
            last_update = elapsed_centis;
            //dump_analysis(static_cast<int>(m_playouts));
			dump_stats(m_rootstate, *m_root);
			if (cfg_puct != 0.8 || cfg_fpu_reduction != 0.25) {
				myprintf("\nThinking... cfg_PUCT: %.2f -> cfg_FPU_reduction: %.2f\n", cfg_puct, cfg_fpu_reduction);
			}
			else {
				myprintf("Thinking...\n");
			}
        }
        keeprunning  = is_running();
		//if ((m_playouts * 100.0) / (elapsed_centis + 1) > 2000) {
		//	return test;
		//}
		Time elapsed4;
		int elapsed_centis4 = Time::timediff_centis(start, elapsed4);
        keeprunning &= !stop_thinking(elapsed_centis4, time_for_move, static_cast<int>(m_playouts));
        if (keeprunning && cfg_timemanage == TimeManagement::ON && elapsed_centis4 - last_update4 > 99) {
			last_update4 = elapsed_centis4;
			//myprintf("\n\n\nLAST_UPDATE4_WAS_CALLED.\n\n\n");
            if (prune_noncontenders(elapsed_centis4, time_for_move, original_time_for_move) == m_root->get_children().size() - 1) {
                myprintf("Stopped early:  %.1fs left\n", (((time_for_move + added_time) - elapsed_centis4)/100.0f));
                //myprintf("Stopping early.\n");
                keeprunning = false;
            }
        }
    } while(keeprunning);

    // reactivate all pruned root children
    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    // stop the search
    m_run = false;
    tg.wait_all();
    m_rootstate.stop_clock(color);
    if (!m_root->has_children()) {
        return FastBoard::PASS;
    }

    // display search info
    myprintf("\n");

    dump_stats(m_rootstate, *m_root);
    Training::record(m_rootstate, *m_root);

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    if (elapsed_centis+1 > 0) {
        myprintf("%d visits, %d nodes, %d playouts, %.0f n/s\n\n",
                 m_root->get_visits(),
                 static_cast<int>(m_nodes),
                 static_cast<int>(m_playouts),
                 (m_playouts * 100.0) / (elapsed_centis+1));
    }
    int bestmove = get_best_move(passflag);

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return bestmove;
}

void UCTSearch::ponder() {
	update_root();

	//m_root->prepare_root_node(m_rootstate.board.get_to_move(), m_nodes, m_rootstate);

	m_run = true;
	ThreadGroup tg(thread_pool);
	for (int i = 1; i < cfg_num_threads; i++) {
		tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
	}
	auto keeprunning = true;
	Time start;                                                     // lizzie
	int last_update = 0;                                            // lizzie    
	do {
		auto currstate = std::make_unique<GameState>(m_rootstate);
		auto result = play_simulation(*currstate, m_root.get());
		if (result.valid()) {
			increment_playouts();
		}
		keeprunning = is_running();
		keeprunning &= !stop_thinking(0, 1);
		Time elapsed;                                               // lizzie
		int elapsed_centis = Time::timediff_centis(start, elapsed); // lizzie
		if (elapsed_centis - last_update > 16) {					// lizzie: output ponder data 6 times per second
			last_update = elapsed_centis;                           // lizzie

			myprintf("~begin\n");                                   // lizzie
			dump_stats(m_rootstate, *m_root);                       // lizzie
			myprintf("~end\n");                                     // lizzie
		}                                                           // lizzie  
	} while (!Utils::input_pending() && keeprunning);

	// stop the search
	m_run = false;
	tg.wait_all();

	// display search info
	myprintf("\n");
	dump_stats(m_rootstate, *m_root);

	myprintf("\n%d visits, %d nodes\n\n", m_root->get_visits(), m_nodes.load());

	// Copy the root state. Use to check for tree re-use in future calls.
	m_last_rootstate = std::make_unique<GameState>(m_rootstate);
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts),
                                      decltype(m_maxplayouts)>::value,
                  "Inconsistent types for playout amount.");
    if (playouts == 0) {
        // Divide max by 2 to prevent overflow when multithreading.
        m_maxplayouts = std::numeric_limits<decltype(m_maxplayouts)>::max()
                        / 2;
    } else {
        m_maxplayouts = playouts;
    }
}

void UCTSearch::set_visit_limit(int visits) {
    static_assert(std::is_convertible<decltype(visits),
                                      decltype(m_maxvisits)>::value,
                  "Inconsistent types for visits amount.");
    if (visits == 0) {
        // Divide max by 2 to prevent overflow when multithreading.
        m_maxvisits = std::numeric_limits<decltype(m_maxvisits)>::max()
                      / 2;
    } else {
        m_maxvisits = visits;
    }
}