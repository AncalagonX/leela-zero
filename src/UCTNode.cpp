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
#include <random>
#include <boost/math/distributions/binomial.hpp>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Utils.h"

using namespace Utils;
using namespace boost::math;

std::random_device rd;

std::mt19937 gen(rd());

std::uniform_int_distribution<> dis4(1, 4);
std::uniform_int_distribution<> dis6(1, 6);
std::uniform_int_distribution<> dis8(1, 8);
std::uniform_int_distribution<> dis10(1, 10);
std::uniform_int_distribution<> dis12(1, 12);
std::uniform_int_distribution<> dis14(1, 14);
std::uniform_int_distribution<> dis16(1, 16);
std::uniform_int_distribution<> dis24(1, 24);
std::uniform_int_distribution<> dis32(1, 32);

int visit_limit_tracking = 1; // This is necessary to properly allocate visits when the user changes search width on the fly. It's set to 1 to avoid any future division-by-zero errors.
int m_visits_tracked_here = 0;

UCTNode::UCTNode(int vertex, float policy) : m_move(vertex), m_policy(policy) {
}

bool UCTNode::first_visit() const {
	if (m_visits == 0) {
		m_visits_tracked_here = 0;
	}
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
    nodelist.emplace_back(raw_netlist.policy_pass, FastBoard::PASS);
    legal_sum += raw_netlist.policy_pass;

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

// Use CI_ALPHA / 2 if calculating double sided bounds.
float UCTNode::get_lcb_binomial(int color) const {
    return get_visits() ? binomial_distribution<>::find_lower_bound_on_p( get_visits(), get_raw_eval(color) * get_visits(), CI_ALPHA) : 0.0f;
}

// Use CI_ALPHA / 2 if calculating double sided bounds.
float UCTNode::get_ucb_binomial(int color) const {
    return get_visits() ? binomial_distribution<>::find_upper_bound_on_p( get_visits(), get_raw_eval(color) * get_visits(), CI_ALPHA) : 1.0f;
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, double(eval));
}

float UCTNode::get_search_width() {
	return m_search_width;
}

void UCTNode::widen_search() {
	m_search_width = (0.558 * m_search_width); // Smaller values cause the search to WIDEN
	if (m_search_width < 0.003) {
		m_search_width = 0.003; // Numbers smaller than (1 / 362) = 0.00276 are theoretically meaningless, but I'll clamp at 100x less than that for now just in case.
		// Update: 0.0000276 crashed leelaz.exe, so I will clamp at 0.00278 which is slightly higher than theoretical minimum.
		// Update2: 0.00278 also crashed, so I'll try clamping at 0.003 instead.
	}
	visit_limit_tracking = (1 + m_visits_tracked_here); // This resets the visit counts used by search limiter. It's necessary to properly allocate visits when the user changes search width on the fly. It's set to 1 to avoid any future division-by-zero errors.
}

void UCTNode::narrow_search() {
	m_search_width = (1.788 * m_search_width); // Larger values cause search to NARROW
	if (m_search_width > 1.0) {
		m_search_width = 1.0; // Numbers larger than 1.0 are meaningless. Clamp to max narrowness of 1.0, which should be identical to traditional LZ search.
	}
	visit_limit_tracking = (1 + m_visits_tracked_here); // This resets the visit counts used by search limiter. It's necessary to properly allocate visits when the user changes search width on the fly. It's set to 1 to avoid any future division-by-zero errors.
}

UCTNode* UCTNode::uct_select_child(int color, bool is_root, int movenum_now, int depth) {
	//LOCK(get_mutex(), lock);
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
	auto best_winrate = std::numeric_limits<double>::lowest();
	auto best_psa = std::numeric_limits<double>::lowest();
	int most_root_visits_seen_so_far = 0;

    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }

		auto winrate = fpu_eval;
		auto lcb_winrate = fpu_eval;
        if (child.is_inflated() && child->m_expand_state.load() == ExpandState::EXPANDING) {
            // Someone else is expanding this node, never select it
            // if we can avoid so, because we'd block on it.
            winrate = -1.0f - fpu_reduction;
        } else if (child.get_visits() > 0) {
            winrate = child.get_eval(color);
			lcb_winrate = child.get_lcb_binomial(color);
        }
		float search_width = get_search_width();

        auto psa = child.get_policy();
        auto denom = 1.0 + child.get_visits();
        auto puct = cfg_puct * psa * (numerator / denom);
		
		if (psa > best_psa) {
			best_psa = psa;
		}
				
		auto value = winrate + puct;

        assert(value > std::numeric_limits<double>::lowest());

		// int randomX = dis8(gen); // UNUSED NOW
		int randomX = dis32(gen);
		int int_m_visits = static_cast<int>(m_visits);
		int int_child_visits = static_cast<int>(child.get_visits());
		int int_parent_visits = static_cast<int>(parentvisits);
		
		if (is_root && depth == 0 && (int_child_visits > most_root_visits_seen_so_far)) {
			most_root_visits_seen_so_far = int_child_visits;
		}

		if (is_root && depth == 0 && (int_m_visits > m_visits_tracked_here)) {
			m_visits_tracked_here = int_m_visits;
		}

		if (randomX <= 2) { // This is a safety catch that prevents an infinite loop race condition that can happen periodically
			if (value > best_value) {
				best_value = value;
				best = &child;
			}
			randomX = dis32(gen);
		}

		/**
		if (is_root
			&& (m_search_width < 1.1
				&& int_child_visits < 2)) {
			if (value > ((0.1 * best_value) + 0.01)) {
				if (value > best_value) {
					best_value = value;
				}
				best = &child;
				assert(best != nullptr);
				best->inflate();
				return best->get();
			}
		}

		if (is_root
			&& (m_search_width < 0.06
				&& int_child_visits < 4)) {
			if (value > ((0.1 * best_value) + 0.01)) {
				if (value > best_value) {
					best_value = value;
				}
				best = &child;
				assert(best != nullptr);
				best->inflate();
				return best->get();
			}
		}
		**/






		/////////////////////////////////////////////////////////////////
		/****************************************************************
		/////////////////////////////////////////////////////////////////

		if (is_root
			&& (m_search_width < 0.01
				&& int_child_visits < 1)) {
			if (puct > 0.001) {
				if (value > best_value) {
					best_value = value;
				}
				best = &child;
				assert(best != nullptr);
				best->inflate();
				return best->get();
			}
		}

		if (is_root
			&& (randomX > 16)
			&& (int_m_visits > 1000)
			&& ((static_cast<float>(most_root_visits_seen_so_far) / static_cast<float>(int_m_visits)) >= 0.25f)
			&& (psa >= 0.1)
			&& (most_root_visits_seen_so_far >= ((1 / psa) + 1))
			&& (int_child_visits < ((10 * (static_cast<int>(most_root_visits_seen_so_far / 1000))) + 398))) {
			if (value > best_value) {
				best_value = value;
				best = &child;
				assert(best != nullptr);
				best->inflate();
				return best->get();
			}
		}

		if (is_root
			&& (randomX > 20)
			&& (int_m_visits > 1000)
			&& ((static_cast<float>(most_root_visits_seen_so_far) / static_cast<float>(int_m_visits)) >= 0.25f)
			&& (psa >= 0.01)
			&& (most_root_visits_seen_so_far >= ((1 / psa) + 1))
			&& (int_child_visits < ((10 * (static_cast<int>(most_root_visits_seen_so_far / 1000))) + 98))) {
			if (value > best_value) {
				best_value = value;
				best = &child;
				assert(best != nullptr);
				best->inflate();
				return best->get();
			}
		}

		if (is_root
			&& (randomX > 24)
			&& (int_m_visits > 1000)
			&& ((static_cast<float>(most_root_visits_seen_so_far) / static_cast<float>(int_m_visits)) >= 0.25f)
			&& (psa >= 0.001)
			&& (most_root_visits_seen_so_far >= ((1 / psa) + 1))
			&& (int_child_visits < ((10 * (static_cast<int>(most_root_visits_seen_so_far / 1000))) + 8))) {
			if (value > best_value) {
				best_value = value;
				best = &child;
				assert(best != nullptr);
				best->inflate();
				return best->get();
			}
		}

		

		//if (is_root
		//	// && int_m_visits > 800 // Allow us to get an instant, unmodified LZ search result 800 visits deep. This allows us to know LZ's unmodified preferred choice immediately.
		//	&& (search_width < 0.99)
		//	&& (randomX <= 30)
		//	&& int_child_visits >= (most_root_visits_seen_so_far - 4)) { // Forces LZ to limit max child visits per root node to a certain ratio of total visits so far. LZ still chooses moves according to its regular "value = winrate + puct" calculation--we simply force it to spend visits on a wider selection of its top move choices.
		//	randomX = dis32(gen);
		//	continue;
		//}

		

		//&& (int_child_visits < ((10 * (static_cast<int>(most_root_visits_seen_so_far / 4000))) + 8))) {

		/////////////////////////////////////////////////////////////////
		****************************************************************/
		/////////////////////////////////////////////////////////////////




		if (is_root && depth == 0
			&& (m_search_width < 0.9
			&& (int_child_visits < 100)
			&& (psa > (0.1 * best_psa))
			&& (value >= (0.95 * best_value)))) {
			
			if (value > best_value) {
				best_value = value;
			}
			best = &child;
			assert(best != nullptr);
			best->inflate();
			return best->get();
		}

		if (is_root && depth == 0
			&& (m_search_width < 0.9
			&& (int_child_visits < 10)
			&& (psa > (0.01 * best_psa))
			&& (value >= (0.95 * best_value)))) {
			
			if (value > best_value) {
				best_value = value;
			}
			best = &child;
			assert(best != nullptr);
			best->inflate();
			return best->get();
		}







		if (is_root && depth == 0
			// && int_m_visits > 800 // Allow us to get an instant, unmodified LZ search result 800 visits deep. This allows us to know LZ's unmodified preferred choice immediately.
			&& (search_width < 0.99)
			&& (static_cast<int>(int_child_visits) >= (static_cast<int>(search_width * (int_m_visits - visit_limit_tracking)) + 10))) { // Forces LZ to limit max child visits per root node to a certain ratio of total visits so far. LZ still chooses moves according to its regular "value = winrate + puct" calculation--we simply force it to spend visits on a wider selection of its top move choices.
			continue;
		}

		if (is_root && depth == 0
			// && int_m_visits > 800 // Allow us to get an instant, unmodified LZ search result 800 visits deep. This allows us to know LZ's unmodified preferred choice immediately.
			&& (search_width > 0.99)
			&& (int_child_visits > (0.95 * int_m_visits))) { // Forces LZ to limit max child visits per root node to a certain ratio of total visits so far. LZ still chooses moves according to its regular "value = winrate + puct" calculation--we simply force it to spend visits on a wider selection of its top move choices.
			continue;
		}

        if (value > best_value) {
			best_value = value;
            best = &child;
        }
		if (winrate > best_winrate) {
			best_winrate = winrate;
		}
    }

    assert(best != nullptr);
	// int randomX = dis8(gen); // UNUSED NOW
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
        // Calculate the lower confidence bound for each node.
		int a_visit = a.get_visits();
		int b_visit = b.get_visits();
        if (a.get_visits() && b.get_visits()) {
            float a_lb = a.get_lcb_binomial(m_color);
            float b_lb = b.get_lcb_binomial(m_color);

            // Sort on lower confidence bounds
            if (a_lb != b_lb) {
                return a_lb < b_lb;
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

