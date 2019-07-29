/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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
#include "FastState.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "FastBoard.h"
#include "Utils.h"
#include "Zobrist.h"
#include "GTP.h"

using namespace Utils;

void FastState::init_game(int size, float komi) {
    board.reset_board(size);

    m_movenum = 0;

    m_komove = FastBoard::NO_VERTEX;
    m_lastmove = FastBoard::NO_VERTEX;
    m_komi = komi;
    m_handicap = 0;
    m_passes = 0;

    return;
}

void FastState::set_komi(float komi) {
    m_komi = komi;
}

void FastState::reset_game() {
    reset_board();

    m_movenum = 0;
    m_passes = 0;
    m_handicap = 0;
    m_komove = FastBoard::NO_VERTEX;
    m_lastmove = FastBoard::NO_VERTEX;
}

void FastState::reset_board() {
    board.reset_board(board.get_boardsize());
}

bool FastState::is_move_legal(int color, int vertex) const {
    return !cfg_analyze_tags.is_to_avoid(color, vertex, m_movenum) && (
              vertex == FastBoard::PASS ||
                 vertex == FastBoard::RESIGN ||
                 (vertex != m_komove &&
                      board.get_state(vertex) == FastBoard::EMPTY &&
                      !board.is_suicide(vertex, color)));
}

bool FastState::is_move_keima(int color, int vertex) const {
    if (is_move_legal(color, vertex)) {

        //int x = vertex % 21;
        //int y = (vertex - x) / 21;

        int x = (vertex % 21) - 1;
        int y = (vertex / 21) - 1;

        int keima1x = x + 1;
        int keima1y = y + 2;

        int keima2x = x + 2;
        int keima2y = y + 1;

        int keima3x = x + 2;
        int keima3y = y - 1;

        int keima4x = x + 1;
        int keima4y = y - 2;

        int keima5x = x - 1;
        int keima5y = y - 2;

        int keima6x = x - 2;
        int keima6y = y - 1;

        int keima7x = x - 2;
        int keima7y = y + 1;

        int keima8x = x - 1;
        int keima8y = y + 2;

        int keima1_vertex = 0;
        int keima2_vertex = 0;
        int keima3_vertex = 0;
        int keima4_vertex = 0;
        int keima5_vertex = 0;
        int keima6_vertex = 0;
        int keima7_vertex = 0;
        int keima8_vertex = 0;

        if (keima1x >= 0 && keima1x < 19 && keima1y >= 0 && keima1y < 19) {
            keima1_vertex = board.get_vertex(keima1x, keima1y);
        }
        if (keima2x >= 0 && keima2x < 19 && keima2y >= 0 && keima2y < 19) {
            keima2_vertex = board.get_vertex(keima2x, keima2y);
        }
        if (keima3x >= 0 && keima3x < 19 && keima3y >= 0 && keima3y < 19) {
            keima3_vertex = board.get_vertex(keima3x, keima3y);
        }
        if (keima4x >= 0 && keima4x < 19 && keima4y >= 0 && keima4y < 19) {
            keima4_vertex = board.get_vertex(keima4x, keima4y);
        }
        if (keima5x >= 0 && keima5x < 19 && keima5y >= 0 && keima5y < 19) {
            keima5_vertex = board.get_vertex(keima5x, keima5y);
        }
        if (keima6x >= 0 && keima6x < 19 && keima6y >= 0 && keima6y < 19) {
            keima6_vertex = board.get_vertex(keima6x, keima6y);
        }
        if (keima7x >= 0 && keima7x < 19 && keima7y >= 0 && keima7y < 19) {
            keima7_vertex = board.get_vertex(keima7x, keima7y);
        }
        if (keima8x >= 0 && keima8x < 19 && keima8y >= 0 && keima8y < 19) {
            keima8_vertex = board.get_vertex(keima8x, keima8y);
        }

        //BLACK = 0, WHITE = 1, EMPTY = 2, INVAL = 3


        bool keima1_bool = false;
        bool keima2_bool = false;
        bool keima3_bool = false;
        bool keima4_bool = false;
        bool keima5_bool = false;
        bool keima6_bool = false;
        bool keima7_bool = false;
        bool keima8_bool = false;

        if (board.get_state(keima1_vertex) == color) {
            keima1_bool = true;
        }
        if (board.get_state(keima2_vertex) == color) {
            keima2_bool = true;
        }
        if (board.get_state(keima3_vertex) == color) {
            keima3_bool = true;
        }
        if (board.get_state(keima4_vertex) == color) {
            keima4_bool = true;
        }
        if (board.get_state(keima5_vertex) == color) {
            keima5_bool = true;
        }
        if (board.get_state(keima6_vertex) == color) {
            keima6_bool = true;
        }
        if (board.get_state(keima7_vertex) == color) {
            keima7_bool = true;
        }
        if (board.get_state(keima8_vertex) == color) {
            keima8_bool = true;
        }
        if (keima1_bool || keima2_bool || keima3_bool || keima4_bool || keima5_bool || keima6_bool || keima7_bool || keima8_bool) {
            return true;
        }
        else {
            return false;
        }
    }
    return false;
}

void FastState::play_move(int vertex) {
    play_move(board.m_tomove, vertex);
}

void FastState::play_move(int color, int vertex) {
    board.m_hash ^= Zobrist::zobrist_ko[m_komove];
    if (vertex == FastBoard::PASS) {
        // No Ko move
        m_komove = FastBoard::NO_VERTEX;
    } else {
        m_komove = board.update_board(color, vertex);
    }
    board.m_hash ^= Zobrist::zobrist_ko[m_komove];

    m_lastmove = vertex;
    m_movenum++;

    if (board.m_tomove == color) {
        board.m_hash ^= Zobrist::zobrist_blacktomove;
    }
    board.m_tomove = !color;

    board.m_hash ^= Zobrist::zobrist_pass[get_passes()];
    if (vertex == FastBoard::PASS) {
        increment_passes();
    } else {
        set_passes(0);
    }
    board.m_hash ^= Zobrist::zobrist_pass[get_passes()];
}

size_t FastState::get_movenum() const {
    return m_movenum;
}

int FastState::get_last_move() const {
    return m_lastmove;
}

int FastState::get_passes() const {
    return m_passes;
}

void FastState::set_passes(int val) {
    m_passes = val;
}

void FastState::increment_passes() {
    m_passes++;
    if (m_passes > 4) m_passes = 4;
}

int FastState::get_to_move() const {
    return board.m_tomove;
}

void FastState::set_to_move(int tom) {
    board.set_to_move(tom);
}

void FastState::display_state() {
    myprintf("\nPasses: %d            Black (X) Prisoners: %d\n",
             m_passes, board.get_prisoners(FastBoard::BLACK));
    if (board.black_to_move()) {
        myprintf("Black (X) to move");
    } else {
        myprintf("White (O) to move");
    }
    myprintf("    White (O) Prisoners: %d\n",
             board.get_prisoners(FastBoard::WHITE));

    board.display_board(get_last_move());
}

std::string FastState::move_to_text(int move) {
    return board.move_to_text(move);
}

float FastState::final_score() const {
    return board.area_score(get_komi() + get_handicap());
}

float FastState::get_komi() const {
    return m_komi;
}

void FastState::set_handicap(int hcap) {
    m_handicap = hcap;
}

int FastState::get_handicap() const {
    return m_handicap;
}

std::uint64_t FastState::get_symmetry_hash(int symmetry) const {
    return board.calc_symmetry_hash(m_komove, symmetry);
}
