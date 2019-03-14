/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto

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

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
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
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Utils.h"
#include "UCTSearch.h"

using namespace Utils;

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
    m_net_eval = raw_netlist.winrate;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    if (state.board.white_to_move()) {
        m_net_eval = 1.0f - m_net_eval;
    }
    eval = m_net_eval;

    std::vector<Network::PolicyVertexPair> nodelist;

    auto legal_sum = 0.0f;
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state.board.get_vertex(x, y);
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(raw_netlist.policy[i], vertex);
            legal_sum += raw_netlist.policy[i];
        }
    }
    // nodelist.emplace_back(raw_netlist.policy_pass, FastBoard::PASS); // ORIGINAL LINE. I commented it out for the below "double pass pathology" PR.
    // legal_sum += raw_netlist.policy_pass; // ORIGINAL LINE. I commented it out for the below "double pass pathology" PR.

	// Add pass move. BUT don't do this if the following conditions
	// all obtain (see issue #2273, "Double-passing pathology"):
	//   - The move played in order to reach this node was a pass.
	//     (So another pass would end the game.)
	//   - The NN's evaluation of the current node is very good
	//     for the player whose move it is. (Say, 0.75 or better.)
	//     (So we don't want to end the game unless we win.)
	//   - Ending the game now would actually lose the game for
	//     the player whose move it is.
	//     (So we don't want to end the game.)
	//   - We do have at least five other legal moves.
	//     (So it's not likely that all our available moves
	//     are actually disastrous.)
	//   - The "dumbpass" option is not turned on.
	//     (Because, as per GCP's comment at
	//     https://github.com/leela-zero/leela-zero/issues/2273#issuecomment-472398802 ,
	//     enabling this heuristic is un-Zero-like and enabling
	//     dumbpass is meant to suppress any such things that
	//     affect passing.)
	// The magic numbers 0.75 and 5 are somewhat arbitrary and it seems
	// unlikely that their values make much difference.
	// This check prevents some serious evaluation errors but has a cost:
	// we make extra calls to final_score() at some nodes. But this is done
	// only at nodes where the other player just passed despite having a
	// really bad position; the cost should not be large.
	if (state.get_passes() == 0
		|| (to_move == FastBoard::WHITE ? 1.0f - m_net_eval : m_net_eval) < 0.75
		|| nodelist.size() < 5
		|| cfg_dumbpass
		|| (to_move == FastBoard::WHITE ? -state.final_score() : state.final_score()) > 0) {
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
    m_visits++;
    accumulate_eval(eval);
}

bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}

bool UCTNode::expandable(const float min_psa_ratio) const {
#ifndef NDEBUG
    if (m_min_psa_ratio_children == 0.0f) {
        // If we figured out that we are fully expandable
        // it is impossible that we stay in INITIAL state.
        assert(m_expand_state.load() != ExpandState::INITIAL);
    }
#endif
    return min_psa_ratio < m_min_psa_ratio_children;
}

float UCTNode::get_policy() const {
    return m_policy;
}

void UCTNode::set_policy(float policy) {
    m_policy = policy;
}

int UCTNode::get_visits() const {
    return m_visits;
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

UCTNode* UCTNode::uct_select_child(int color, bool is_root, int movenum_here, bool is_depth_1, bool is_opponent_move, bool is_pondering_now) {
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

    const auto numerator = std::sqrt(double(parentvisits) *
            std::log(cfg_logpuct * double(parentvisits) + cfg_logconst));
    const auto fpu_reduction = (is_root ? cfg_fpu_root_reduction : cfg_fpu_reduction) * std::sqrt(total_visited_policy);
    // Estimated eval for unknown nodes = original parent NN eval - reduction
    const auto fpu_eval = get_net_eval(color) - fpu_reduction;

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value = std::numeric_limits<double>::lowest();
	auto best_root_winrate = std::numeric_limits<double>::lowest();

    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }

        auto winrate = fpu_eval;
        if (child.is_inflated() && child->m_expand_state.load() == ExpandState::EXPANDING) {
            // Someone else is expanding this node, never select it
            // if we can avoid so, because we'd block on it.
            winrate = -1.0f - fpu_reduction;
        } else if (child.get_visits() > 0) {
            winrate = child.get_eval(color);
        }
        const auto psa = child.get_policy();
        const auto denom = 1.0 + child.get_visits();
        const auto puct = cfg_puct * psa * (numerator / denom);
        const auto value = winrate + puct;
        assert(value > std::numeric_limits<double>::lowest());

		int int_child_visits = static_cast<int>(child.get_visits());

		if ((is_root | is_depth_1)
			&& (!is_opponent_move)) {
			if (winrate > best_root_winrate) {
				best_root_winrate = winrate;
			}
		}



		// FAST FIRST 8 MOVES
		if (is_root
			&& (movenum_here < 8)) {
			UCTSearch::set_playout_limit(3200);
		}

		if (is_root
			&& (movenum_here >= 8) && (movenum_here < 50)) {
			UCTSearch::set_playout_limit(UCTSearch::UNLIMITED_PLAYOUTS);
		}



		// CONTROL TIME MANAGEMENT IF NOT WINNING QUICKLY
		if (is_root
			&& (movenum_here < 20)) {
			cfg_timemanage = TimeManagement::FAST;
		}

		if (is_root
			&& (movenum_here >= 20)	&& (movenum_here < 40)
			&& (best_root_winrate <= 0.65)
			&& (int_child_visits >= 1600)) {
			cfg_timemanage = TimeManagement::OFF;
		}
		if (is_root
			&& (movenum_here >= 20)	&& (movenum_here < 40)
			&& (best_root_winrate >= 0.66)
			&& (int_child_visits >= 1600)) {
			cfg_timemanage = TimeManagement::FAST;
		}

		if (is_root
			&& (movenum_here >= 40) && (movenum_here < 150)
			&& (best_root_winrate <= 0.8)
			&& (int_child_visits >= 800)) {
			cfg_timemanage = TimeManagement::OFF;
		}
		if (is_root
			&& (movenum_here >= 40) && (movenum_here < 150)
			&& (best_root_winrate >= 0.81)
			&& (int_child_visits >= 800)) {
			cfg_timemanage = TimeManagement::FAST;
		}

		if (is_root
			&& (movenum_here >= 150)
			&& (best_root_winrate <= 0.9)
			&& (int_child_visits >= 800)) {
			cfg_timemanage = TimeManagement::OFF;
		}
		if (is_root
			&& (movenum_here >= 150)
			&& (best_root_winrate >= 0.91)
			&& (int_child_visits >= 800)) {
			cfg_timemanage = TimeManagement::FAST;
		}



		// CONTROL PLAYOUT LIMIT IF EXTREMELY WINNING
		if (is_root
			&& (movenum_here >= 50)
			&& (best_root_winrate >= 0.90)
			&& (int_child_visits >= 400)) {
			UCTSearch::set_playout_limit(1600);
		}
		if (is_root
			&& (movenum_here >= 50)
			&& (best_root_winrate <= 0.89)
			&& (int_child_visits >= 400)) {
			UCTSearch::set_playout_limit(UCTSearch::UNLIMITED_PLAYOUTS);
		}

		if (is_root
			&& (movenum_here >= 100)
			&& (best_root_winrate >= 0.90)
			&& (int_child_visits >= 400)) {
			UCTSearch::set_playout_limit(1200);
		}
		if (is_root
			&& (movenum_here >= 100)
			&& (best_root_winrate <= 0.89)
			&& (int_child_visits >= 400)) {
			UCTSearch::set_playout_limit(UCTSearch::UNLIMITED_PLAYOUTS);
		}

		if (is_root
			&& (movenum_here >= 150)
			&& (best_root_winrate >= 0.90)
			&& (int_child_visits >= 400)) {
			UCTSearch::set_playout_limit(800);
		}
		if (is_root
			&& (movenum_here >= 150)
			&& (best_root_winrate <= 0.89)
			&& (int_child_visits >= 400)) {
			UCTSearch::set_playout_limit(UCTSearch::UNLIMITED_PLAYOUTS);
		}

        if (value > best_value) {
            best_value = value;
            best = &child;
        }
    }

    assert(best != nullptr);
    best->inflate();
    return best->get();
}

class NodeComp : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    NodeComp(int color) : m_color(color) {};

    // WARNING : on very unusual cases this can be called on multithread
    // contexts (e.g., UCTSearch::get_pv()) so beware of race conditions
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        auto a_visit = a.get_visits();
        auto b_visit = b.get_visits();

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
};

void UCTNode::sort_children(int color) {
    std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color));
}

UCTNode& UCTNode::get_best_root_child(int color) {
    wait_expanded();

    assert(!m_children.empty());

    auto ret = std::max_element(begin(m_children), end(m_children),
                                NodeComp(color));
    ret->inflate();

    return *(ret->get());
}

size_t UCTNode::count_nodes_and_clear_expand_state() {
    auto nodecount = size_t{0};
    nodecount += m_children.size();
    if (expandable()) {
        m_expand_state = ExpandState::INITIAL;
    }
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

