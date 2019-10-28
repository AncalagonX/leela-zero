/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto

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

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Utils.h"

using namespace Utils;
int most_root_visits_seen = 0;
int second_most_root_visits_seen = 0;
int vertex_most_root_visits_seen = 0;
int vertex_second_most_root_visits_seen = 0;
float best_root_winrate = 0.0f;

std::random_device rd;

std::mt19937 gen(rd());

std::uniform_int_distribution<> dis100(1, 100);
std::uniform_real_distribution<> dis_float_fuzz_zero_to_two(0.01, 1.99);
std::uniform_real_distribution<> dis_float_fuzz_0_5_to_1_5(0.50, 1.50);

UCTNode::UCTNode(int vertex, float policy) : m_move(vertex), m_policy(policy) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

bool UCTNode::create_children(Network & network,
                              std::atomic<int>& nodecount,
                              GameState& state,
                              float& eval,
                              float min_psa_ratio) {
    // no successors in final state
    if (state.get_passes() >= 2) {
        return false;
    }

    // acquire the lock
    if (!acquire_expanding()) {
        return false;
    }

    // can we actually expand?
    if (!expandable(min_psa_ratio)) {
        expand_done();
        return false;
    }

    const auto raw_netlist = network.get_output(
        &state, Network::Ensemble::RANDOM_SYMMETRY);

    // DCNN returns winrate as side to move
    const auto stm_eval = raw_netlist.winrate;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    if (to_move == FastBoard::WHITE) {
        m_net_eval = 1.0f - stm_eval;
    } else {
        m_net_eval = stm_eval;
    }
    eval = m_net_eval;

    std::vector<Network::PolicyVertexPair> nodelist;

    auto legal_sum = 0.0f;
    for (auto i = 0; i < BOARD_SQUARES; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state.board.get_vertex(x, y);
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(raw_netlist.policy[i], vertex);
            legal_sum += raw_netlist.policy[i];
        }
    }

    // Always try passes if we're not trying to be clever.
    auto allow_pass = cfg_dumbpass;

    // Less than 20 available intersections in a 19x19 game.
    if (nodelist.size() <= std::max(5, BOARD_SIZE)) {
        allow_pass = true;
    }

    // If we're clever, only try passing if we're winning on the
    // net score and on the board count.
    if (!allow_pass && stm_eval > 0.8f) {
        const auto relative_score =
            (to_move == FastBoard::BLACK ? 1 : -1) * state.final_score();
        if (relative_score >= 0) {
            allow_pass = true;
        }
    }

    allow_pass = true;

    if (allow_pass) {
        nodelist.emplace_back(raw_netlist.policy_pass, FastBoard::PASS);
        legal_sum += raw_netlist.policy_pass;
    }

    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    } else {
        // This can happen with new randomized nets.
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) {
            node.first = uniform_prob;
        }
    }

    link_nodelist(nodecount, nodelist, min_psa_ratio);
    expand_done();
    return true;
}

void UCTNode::link_nodelist(std::atomic<int>& nodecount,
                            std::vector<Network::PolicyVertexPair>& nodelist,
                            float min_psa_ratio) {
    assert(min_psa_ratio < m_min_psa_ratio_children);

    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    const auto max_psa = nodelist[0].first;
    const auto old_min_psa = max_psa * m_min_psa_ratio_children;
    const auto new_min_psa = max_psa * min_psa_ratio;
    if (new_min_psa > 0.0f) {
        m_children.reserve(
            std::count_if(cbegin(nodelist), cend(nodelist),
                [=](const auto& node) { return node.first >= new_min_psa; }
            )
        );
    } else {
        m_children.reserve(nodelist.size());
    }

    auto skipped_children = false;
    for (const auto& node : nodelist) {
        if (node.first < new_min_psa) {
            skipped_children = true;
        } else if (node.first < old_min_psa) {
            m_children.emplace_back(node.second, node.first);
            ++nodecount;
        }
    }

    m_min_psa_ratio_children = skipped_children ? min_psa_ratio : 0.0f;
}

const std::vector<UCTNodePointer>& UCTNode::get_children() const {
    return m_children;
}


int UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

void UCTNode::update(float eval) {
    // Cache values to avoid race conditions.
    auto old_eval = static_cast<float>(m_blackevals);
    auto old_visits = static_cast<int>(m_visits);
    auto old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    m_visits++;
    accumulate_eval(eval);
    auto new_delta = eval - (old_eval + eval) / (old_visits + 1);
    // Welford's online algorithm for calculating variance.
    auto delta = old_delta * new_delta;
    atomic_add(m_squared_eval_diff, delta);
}

bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}

bool UCTNode::expandable(const float min_psa_ratio) const {
    return min_psa_ratio < m_min_psa_ratio_children;
}

float UCTNode::get_policy() const {
    return m_policy;
}

void UCTNode::set_policy(float policy) {
    m_policy = policy;
}

float UCTNode::get_eval_variance(float default_var) const {
    return m_visits > 1 ? m_squared_eval_diff / (m_visits - 1) : default_var;
}

int UCTNode::get_visits() const {
    return m_visits;
}

float UCTNode::get_eval_lcb(int color) const {
    // Lower confidence bound of winrate.
    auto visits = get_visits();
    if (visits < 2) {
        // Return large negative value if not enough visits.
        return -1e6f + visits;
    }
    auto mean = get_raw_eval(color);

    auto stddev = std::sqrt(get_eval_variance(1.0f) / visits);
    auto z = cached_t_quantile(visits - 1);

    return mean - z * stddev;
}

float UCTNode::get_raw_eval(int tomove, int virtual_loss) const {
    auto visits = get_visits() + virtual_loss;
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += static_cast<double>(virtual_loss);
    }
    auto eval = static_cast<float>(blackeval / double(visits));
    if (tomove == FastBoard::WHITE) {
        eval = 1.0f - eval;
    }
    return eval;
}

float UCTNode::get_eval(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    return get_raw_eval(tomove, m_virtual_loss);
}

float UCTNode::get_net_eval(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return 1.0f - m_net_eval;
    }
    return m_net_eval;
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, double(eval));
}

//UCTNode* UCTNode::uct_select_child(int color, bool is_root, int movenum_now, bool is_depth_1, bool is_opponent_move, bool is_pondering_now) {
//uct_select_child(int color, int color_to_move, bool is_root, int movenum_now, int depth, bool is_pondering_now)
UCTNode* UCTNode::uct_select_child(int color, int color_to_move, bool is_root, int movenum_now, int depth, bool is_pondering_now) {
    wait_expanded();

    // Count parentvisits manually to avoid issues with transpositions.
    auto total_visited_policy = 0.0f;
    auto parentvisits = size_t{0};
    for (const auto& child : m_children) {
        if (child.valid()) {
            parentvisits += child.get_visits();
            if (child.get_visits() > 0) {
                total_visited_policy += child.get_policy();
            }
        }
    }

    // Random number from [0, max - 1]
    //std::uint64_t randuint64(const std::uint64_t max);

    //std::random_device rd;
    //std::ranlux48 gen(rd());
    //std::uint64_t random_test_integer = (gen() << 16) ^ gen();

    const auto numerator = std::sqrt(double(parentvisits) *
            std::log(cfg_logpuct * double(parentvisits) + cfg_logconst));
    const auto fpu_reduction = (is_root ? cfg_fpu_root_reduction : cfg_fpu_reduction) * std::sqrt(total_visited_policy);

   
    // The two lines below were for me to test displaying the total_visit_policy values:
    //auto current_move_vertex = get_move();
    //myprintf("%d = %.5f total_visited_policy.\n", current_move_vertex, total_visited_policy);



    // Estimated eval for unknown nodes = original parent NN eval - reduction
    const auto fpu_eval = get_net_eval(color) - fpu_reduction;

    //NOTE: I CHANGED THE ABOVE ORIGINAL FPU_EVAL CODE TO JUST BE A FLAT 0.50:
    //const auto fpu_eval = 0.50f;

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value = std::numeric_limits<double>::lowest();
    best_root_winrate = std::numeric_limits<double>::lowest();


    auto second_best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value2 = std::numeric_limits<double>::lowest();
    auto best_value_next = std::numeric_limits<double>::lowest();
    auto best_winrate = std::numeric_limits<double>::lowest();
    auto best_winrate2 = std::numeric_limits<double>::lowest();
    auto best_lcb = std::numeric_limits<double>::lowest();
    auto best_psa = std::numeric_limits<double>::lowest();
    int most_root_visits_seen_so_far = 1;
    int second_most_root_visits_seen_so_far = 1;


    auto winrate_target_value = 0.01f * cfg_winrate_target; // Converts user input into float between 1.0f and 0.0f
    auto raw_winrate_target_value = 0.01f * cfg_winrate_target; // Converts user input into float between 1.0f and 0.0f
    if (movenum_now < 100) {
        winrate_target_value = 0.01f * (cfg_winrate_target); // Converts user input into float between 1.0f and 0.0f
    }
    if (movenum_now >= 100) {
        winrate_target_value = 0.01f * (cfg_winrate_target + 5); // Converts user input into float between 1.0f and 0.0f
    }

    if (movenum_now >= 150) {
        winrate_target_value = 0.01f * (cfg_winrate_target + 10); // Converts user input into float between 1.0f and 0.0f
    }
    if (movenum_now >= 200) {
        winrate_target_value = 0.01f * (cfg_winrate_target + 15); // Converts user input into float between 1.0f and 0.0f
    }


    float movenum_float = movenum_now * 1.0f;



    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }

        auto winrate = fpu_eval;
        if (child.is_inflated() && child->m_expand_state.load() == ExpandState::EXPANDING) {
            // Someone else is expanding this node, never select it
            // if we can avoid so, because we'd block on it.
            winrate = -1.0f - fpu_reduction;
        }
        else if (child.get_visits() > 0) {
            winrate = child.get_eval(color);
        }
        const auto psa = child.get_policy();
        const auto denom = 1.0 + child.get_visits();
        const auto puct = cfg_puct * psa * (numerator / denom);

        /**
        if (movenum_now <= 250) {
            winrate = winrate * (movenum_float / 250.0f);
        }
        **/

        /**

        // "If" statement above replaced with single line below:
        winrate = winrate * std::min(1.0, movenum_now / 1000.0);

        float winrate_fuzz_factor_zero_to_two = dis_float_fuzz_zero_to_two(gen);
        float winrate_fuzz_factor_0_5_to_1_5 = dis_float_fuzz_0_5_to_1_5(gen);

        winrate = winrate * winrate_fuzz_factor_zero_to_two;
        //winrate = winrate * winrate_fuzz_factor_0_5_to_1_5;

    **/

        auto value = winrate + puct;

        bool is_opponent_move = ((depth % 2) != 0); // Returns "true" on moves at odd-numbered depth, indicating at any depth in a search variation which moves are played by LZ's opponent.

        if (is_pondering_now) {
            is_opponent_move = !is_opponent_move; // When white's turn, opponent's moves are made at even-numbered depths. Flipping this bool accounts for this.
        }

        if (is_root && (static_cast<int>(child.get_visits()) > most_root_visits_seen)) {
            if (vertex_most_root_visits_seen != child.get_move()) {
                vertex_most_root_visits_seen = child.get_move();
                second_most_root_visits_seen = most_root_visits_seen;
            }
            most_root_visits_seen = static_cast<int>(child.get_visits());
            if (most_root_visits_seen >= 1) {
                best_root_winrate = winrate;
            }
        }

        if (is_root
            && (static_cast<int>(child.get_visits()) < most_root_visits_seen)
            && (static_cast<int>(child.get_visits()) > second_most_root_visits_seen)) {
            if (vertex_second_most_root_visits_seen != child.get_move()) {
                vertex_second_most_root_visits_seen = child.get_move();
            }
            second_most_root_visits_seen = static_cast<int>(child.get_visits());
        }






        if (cfg_tengen == true) {
            int check_vertex = static_cast<int>(child.get_move());
            int remainder_vertex = check_vertex % 21;
            int leftover_vertex = (check_vertex - remainder_vertex) / 21;

            if ((movenum_now + depth <= 1) && (check_vertex == 221)) { //220 is tengen. 221 is one intersection away.
                value = 1000.0 * value;
            }

            if ((movenum_now + depth <= 1) && (check_vertex != 221)) { //220 is tengen. 221 is one intersection away.
                value = value / 1000.0;
            }
        }

        if (cfg_faster == true) {
            if (is_opponent_move && (depth == 0) && (movenum_now + depth <= 100)) { // wider search during ponder
                value = winrate + (10.0f * puct);
            }
        }

        if (cfg_tengenbot == true) {
        /////////////////////////////////////////////////////////////////////////////////
        // TENGEN-FOCUSED: //
        /////////////////////////////////////////////////////////////////////////////////

            //if (!is_opponent_move && (movenum_now + depth <= 10) && (((movenum_now + depth) % 10) != 8) && (((movenum_now + depth) % 10) != 9) && (winrate >= 0.40)) {
            if (!is_opponent_move && (movenum_now + depth <= 10) && (winrate >= 0.40)) {
                int check_vertex = static_cast<int>(child.get_move());
                int remainder_vertex = check_vertex % 21;
                int leftover_vertex = (check_vertex - remainder_vertex) / 21;
                if (leftover_vertex <= 4 || leftover_vertex >= 16) {
                    value = 0.90 * value;
                }
                if (remainder_vertex <= 4 || remainder_vertex >= 16) {
                    value = 0.90 * value;
                }

                if (leftover_vertex <= 3 || leftover_vertex >= 17) {
                    value = 0.90 * value;
                }
                if (remainder_vertex <= 3 || remainder_vertex >= 17) {
                    value = 0.90 * value;
                }
            }
            
            //if (!is_opponent_move && (movenum_now + depth > 10) && (movenum_now + depth <= 80) && (((movenum_now + depth) % 10) != 8) && (((movenum_now + depth) % 10) != 9) && (winrate >= 0.60)) {
            if (!is_opponent_move && (movenum_now + depth > 10) && (movenum_now + depth <= 80) && (winrate >= 0.60)) {
                int check_vertex = static_cast<int>(child.get_move());
                int remainder_vertex = check_vertex % 21;
                int leftover_vertex = (check_vertex - remainder_vertex) / 21;
                if (leftover_vertex <= 4 || leftover_vertex >= 16) {
                    value = 0.90 * value;
                }
                if (remainder_vertex <= 4 || remainder_vertex >= 16) {
                    value = 0.90 * value;
                }

                if (leftover_vertex <= 3 || leftover_vertex >= 17) {
                    value = 0.95 * value;
                }
                if (remainder_vertex <= 3 || remainder_vertex >= 17) {
                    value = 0.95 * value;
                }

                //if (leftover_vertex <= 2 || leftover_vertex >= 18) {
                //    value = 0.90 * value;
                //}
                //if (remainder_vertex <= 2 || remainder_vertex >= 18) {
                //    value = 0.90 * value;
                //}

                /**
                if (cfg_tengen == true) {
                    if ((movenum_now + depth <= 1) && (check_vertex == 220)) {
                        value = 1000.0 * value;
                    }

                    if ((movenum_now + depth <= 1) && (check_vertex != 220)) {
                        value = value / 1000.0;
                    }
                }
                **/

                // 9x10  = 199
                // 11x10 = 241
                // 10x9  = 219
                // 10x11 = 221

                //if ((movenum_now + depth == 1) && (check_vertex == 220) && (get_move() != 199) && (get_move() != 241) && (get_move() != 219) && (get_move() != 221)) {
                //    value = 1000.0 * value;
                //}
                //if (movenum_now + depth == 1) {
                //    if ((get_move() == 199) || (get_move() == 241) || (get_move() == 219) || (get_move() == 221)) {
                //    }if (check_vertex == 220) {
                //        value = value / 1000.0;
                //    }
                //}
            }

            //if (!is_opponent_move && (movenum_now + depth > 80) && (movenum_now + depth <= 100) && (((movenum_now + depth) % 10) != 8) && (((movenum_now + depth) % 10) != 9) && (winrate >= 0.65)) {
            if (!is_opponent_move && (movenum_now + depth > 80) && (movenum_now + depth <= 100) && (winrate >= 0.65)) {
                int check_vertex = static_cast<int>(child.get_move());
                int remainder_vertex = check_vertex % 21;
                int leftover_vertex = (check_vertex - remainder_vertex) / 21;
                if (leftover_vertex <= 4 || leftover_vertex >= 16) {
                    value = 0.95 * value;
                }
                if (remainder_vertex <= 4 || remainder_vertex >= 16) {
                    value = 0.95 * value;
                }

                if (leftover_vertex <= 3 || leftover_vertex >= 17) {
                    value = 0.95 * value;
                }
                if (remainder_vertex <= 3 || remainder_vertex >= 17) {
                    value = 0.95 * value;
                }

                //if (leftover_vertex <= 2 || leftover_vertex >= 18) {
                //    value = 0.90 * value;
                //}
                //if (remainder_vertex <= 2 || remainder_vertex >= 18) {
                //    value = 0.90 * value;
                //}

                if ((movenum_now + depth <= 1) && (check_vertex == 220)) {
                    value = 1000.0 * value;
                }

                if ((movenum_now + depth <= 1) && (check_vertex != 220)) {
                    value = value / 1000.0;
                }

                // 9x10  = 199
                // 11x10 = 241
                // 10x9  = 219
                // 10x11 = 221

                //if ((movenum_now + depth == 1) && (check_vertex == 220) && (get_move() != 199) && (get_move() != 241) && (get_move() != 219) && (get_move() != 221)) {
                //    value = 1000.0 * value;
                //}
                //if (movenum_now + depth == 1) {
                //    if ((get_move() == 199) || (get_move() == 241) || (get_move() == 219) || (get_move() == 221)) {
                //    }if (check_vertex == 220) {
                //        value = value / 1000.0;
                //    }
                //}
            }
        }




        assert(value > std::numeric_limits<double>::lowest());

        if (value > best_value) {
            best_value = value;
            best = &child;
        }






        if (cfg_passbot == true) {

            int int_m_visits = static_cast<int>(m_visits);
            int int_child_visits = static_cast<int>(child.get_visits());
            int int_parent_visits = static_cast<int>(parentvisits);

            if (is_root && (int_child_visits > most_root_visits_seen_so_far)) {
                second_most_root_visits_seen_so_far = most_root_visits_seen_so_far;
                most_root_visits_seen_so_far = int_child_visits;
            }

            // Ignore considering opponent passing if we just passed (#1)

            if (is_opponent_move
                && (child.get_move() == -1)
                && (movenum_now <= 250)) {
                continue;
            }

            // Having this "if statement" first ensures default LZ search picks a "best move" in the rare case of failure in the "Pass Bot" sections below.

            if (value > best_value) {
                if (!is_opponent_move && !(child.get_move() == -1) && (winrate > best_winrate) && (int_m_visits > 800)) {
                    best_winrate = winrate;
                }
                best_value = value;
                best_value2 = value;
                best = &child;
            }



            // Ignore considering opponent passing if we just passed (#2)

            if (is_opponent_move
                && (get_move() == -1)
                && (child.get_move() == -1)
                && (movenum_now <= 250)) {
                continue;
            }


            // If root and it's our turn, always send 50 visits into "Pass".

            if (!is_opponent_move
                //&& (is_root)
                && (depth <= 1)
                && (movenum_now <= 250)
                && (child.get_move() == -1)
                && (int_child_visits <= 50)) {
                if (value > best_value) {
                    best_value = value;
                }
                best = &child;
                assert(best != nullptr);
                best->inflate();
                return best->get();
            }

            // If root and it's our turn, always send "Pass" by extra visits equal to:  approximately 5% of highest root move visits so far

            /**
            if (!is_opponent_move
                && (is_root)
                && (movenum_now <= 250)
                && (child.get_move() == -1)
                && (int_child_visits <= (50 + (10 * static_cast<int>(0.1 * static_cast<int>(0.05f * most_root_visits_seen_so_far)))))) {
                if (value > best_value) {
                    best_value = value;
                }
                best = &child;
                assert(best != nullptr);
                best->inflate();
                return best->get();
            }
            **/
            

            // If root and it's our turn, AND "Pass" is >= winrate_target_value, send ALL visits to it.

            if (!is_opponent_move
                //&& (is_root)
                && (depth <= 1)
                && (movenum_now <= 250)
                && (child.get_move() == -1)) {
                if (value > best_value) {
                    best_value = value;
                }
                if ((winrate >= winrate_target_value)
                    && (child.get_visits() < (0.60f * int_m_visits))) {
                    best = &child;
                    assert(best != nullptr);
                    best->inflate();
                    return best->get();
                }
            }

            /*****************
            ******* COPIED FROM VERTEX BRANCH CODE
            if (!is_opponent_move && (color_to_move != cfg_opponent) && (depth <= 20)) {
                value = (puct)+((winrate + puct) * ((get_move() == vertex_to_search_for_4a) / (depth + 1)));
            }

            *****************/


        }






    }

    if (cfg_passbot == true && cfg_passbot == false) {
        for (auto& child : m_children) {
            if (!child.active()) {
                continue;
            }

            auto winrate = fpu_eval;
            auto lcb = 0.0f;
            if (child.is_inflated() && child->m_expand_state.load() == ExpandState::EXPANDING) {
                // Someone else is expanding this node, never select it
                // if we can avoid so, because we'd block on it.
                winrate = -1.0f - fpu_reduction;
            }
            else if (child.get_visits() > 0) {
                winrate = child.get_eval(color);

            }
            const auto psa = child.get_policy();
            const auto denom = 1.0 + child.get_visits();
            auto puct = cfg_puct * psa * (numerator / denom);

            int int_m_visits = static_cast<int>(m_visits);
            int int_child_visits = static_cast<int>(child.get_visits());
            int int_parent_visits = static_cast<int>(parentvisits);

            auto value = winrate + puct;

            bool is_opponent_move = ((depth % 2) != 0); // Returns "true" on moves at odd-numbered depth, indicating at any depth in a search variation which moves are played by LZ's opponent.

            if (is_pondering_now) {
                is_opponent_move = !is_opponent_move; // When white's turn, opponent's moves are made at even-numbered depths. Flipping this bool accounts for this.
            }

            /**

            if (!is_opponent_move && (winrate >= winrate_target_value)) {
                value = (1 - abs(winrate_target_value - winrate)) + puct;
            }

            **/

            assert(value > std::numeric_limits<double>::lowest());

            if (is_root && depth == 0 && (int_child_visits > most_root_visits_seen_so_far)) {
                second_most_root_visits_seen_so_far = most_root_visits_seen_so_far;
                most_root_visits_seen_so_far = int_child_visits;
            }

            if ((get_move() == -1
                && (child.get_move() == -1)
                && (movenum_now <= 150))) {
                continue;
            }


            /**
            if (!is_opponent_move
                && (child.get_move() == -1)
                && (int_child_visits == 0)) {
                if (value > best_value) {
                    best_value = value;
                }
                best = &child;
                assert(best != nullptr);
                best->inflate();
                return best->get();
            }
            **/


            if (!is_opponent_move
                && is_root
                && (child.get_move() == -1)
                && (int_child_visits <= 400)) {
                if (value > best_value) {
                    best_value = value;
                }
                best = &child;
                assert(best != nullptr);
                best->inflate();
                return best->get();
            }

            if (!is_opponent_move
                && is_root
                && (child.get_move() == -1)) {
                //&& (int_child_visits >= 400)) {
                if (value > best_value) {
                    best_value = value;
                }
                if (winrate >= winrate_target_value) {
                    best = &child;
                    assert(best != nullptr);
                    best->inflate();
                    return best->get();
                }
            }

            /*****************
            ******* COPIED FROM VERTEX BRANCH CODE
            if (!is_opponent_move && (color_to_move != cfg_opponent) && (depth <= 20)) {
                value = (puct)+((winrate + puct) * ((get_move() == vertex_to_search_for_4a) / (depth + 1)));
            }

            *****************/

            if (value > best_value) {
                if (!is_opponent_move && (winrate > best_winrate) && (int_m_visits > 100)) {
                    best_winrate = winrate;
                }
                best_value = value;
                best_value2 = value;
                best = &child;
            }
        }
    }

    assert(best != nullptr);
    best->inflate();
    return best->get();
}

class NodeComp : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    NodeComp(int color, float lcb_min_visits) : m_color(color),
        m_lcb_min_visits(lcb_min_visits){};

    // WARNING : on very unusual cases this can be called on multithread
    // contexts (e.g., UCTSearch::get_pv()) so beware of race conditions
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        auto a_visit = a.get_visits();
        auto b_visit = b.get_visits();

        // Need at least 2 visits for LCB.
        if (m_lcb_min_visits < 2) {
            m_lcb_min_visits = 2;
        }

        // Calculate the lower confidence bound for each node.
        if ((a_visit > m_lcb_min_visits) && (b_visit > m_lcb_min_visits)) {
            auto a_lcb = a.get_eval_lcb(m_color);
            auto b_lcb = b.get_eval_lcb(m_color);

            // Sort on lower confidence bounds
            if (cfg_passbot == false || cfg_passbot == true) { // Passbot mode should work fine with LCB threshold
                if (a_lcb != b_lcb) {
                    return a_lcb < b_lcb;
                }
            }
        }

        // if visits are not same, sort on visits
        if (a_visit != b_visit) {
            return a_visit < b_visit;
        }

        // neither has visits, sort on policy prior
        if (a_visit == 0) {
            return a.get_policy() < b.get_policy();
        }

        // both have same non-zero number of visits
        return a.get_eval(m_color) < b.get_eval(m_color);
    }
private:
    int m_color;
    float m_lcb_min_visits;
};

void UCTNode::sort_children(int color, float lcb_min_visits) {
    std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color, lcb_min_visits));
}

UCTNode& UCTNode::get_best_root_child(int color) {
    wait_expanded();

    assert(!m_children.empty());

    auto max_visits = 0;
    for (const auto& node : m_children) {
        max_visits = std::max(max_visits, node.get_visits());
    }

    auto ret = std::max_element(begin(m_children), end(m_children),
                                NodeComp(color, cfg_lcb_min_visit_ratio * max_visits));
    ret->inflate();

    return *(ret->get());
}

size_t UCTNode::count_nodes_and_clear_expand_state() {
    auto nodecount = size_t{0};
    nodecount += m_children.size();
    m_expand_state = ExpandState::INITIAL;
    for (auto& child : m_children) {
        if (child.is_inflated()) {
            nodecount += child->count_nodes_and_clear_expand_state();
        }
    }
    return nodecount;
}

void UCTNode::invalidate() {
    m_status = INVALID;
}

void UCTNode::set_active(const bool active) {
    if (valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

bool UCTNode::valid() const {
    return m_status != INVALID;
}

bool UCTNode::active() const {
    return m_status == ACTIVE;
}

bool UCTNode::acquire_expanding() {
    auto expected = ExpandState::INITIAL;
    auto newval = ExpandState::EXPANDING;
    return m_expand_state.compare_exchange_strong(expected, newval);
}

void UCTNode::expand_done() {
    auto v = m_expand_state.exchange(ExpandState::EXPANDED);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::expand_cancel() {
    auto v = m_expand_state.exchange(ExpandState::INITIAL);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::wait_expanded() {
    while (m_expand_state.load() == ExpandState::EXPANDING) {}
    auto v = m_expand_state.load();
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDED);
}

