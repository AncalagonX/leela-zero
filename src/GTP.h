/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

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

#ifndef GTP_H_INCLUDED
#define GTP_H_INCLUDED

#include "config.h"

#include <cstdio>
#include <string>
#include <vector>

#include "Network.h"
#include "GameState.h"
#include "UCTSearch.h"

extern bool cfg_gtp_mode;
extern bool cfg_allow_pondering;
extern bool cfg_passbot;
extern bool cfg_tengenbot;
extern bool cfg_tengen;
extern bool cfg_tengenchat;
extern bool cfg_kageyamachat;
extern bool cfg_hiddenwinrate;
extern bool cfg_capturestones;
extern bool cfg_tiebot;
extern bool cfg_handicapadjustment;
extern bool cfg_handicapgame;
extern bool cfg_nofirstlinemovesearly;
extern bool win_message_sent;
extern bool cfg_faster;
extern bool cfg_superslow;
extern int cfg_winrate_target;
extern int cfg_num_threads;
extern int cfg_max_threads;
extern int cfg_max_playouts;
extern int cfg_max_visits;
extern int cfg_single_move_visit_limit;
extern float cfg_second_best_move_ratio;
extern float cfg_handicapadjustmentpercent;
extern int cfg_single_move_visits_required_to_check;
extern TimeManagement::enabled_t cfg_timemanage;
extern int cfg_lagbuffer_cs;
extern int cfg_resignpct;
extern int cfg_resign_moves;
extern int resign_moves_counter;
extern int cfg_noise;
extern int cfg_random_cnt;
extern int cfg_random_min_visits;
extern float cfg_random_temp;
extern std::uint64_t cfg_rng_seed;
extern bool cfg_dumbpass;
#ifdef USE_OPENCL
extern std::vector<int> cfg_gpus;
extern bool cfg_sgemm_exhaustive;
extern bool cfg_tune_only;
#ifdef USE_HALF
enum class precision_t {
    AUTO, SINGLE, HALF
};
extern precision_t cfg_precision;
#endif
#endif
extern float cfg_puct;
extern float cfg_logpuct;
extern float cfg_logconst;
extern float cfg_softmax_temp;
extern float cfg_fpu_reduction;
extern float cfg_fpu_root_reduction;
extern float cfg_ci_alpha;
extern float cfg_lcb_min_visit_ratio;
extern std::string cfg_logfile;
extern std::string cfg_weightsfile;
extern FILE* cfg_logfile_handle;
extern bool cfg_quiet;
extern std::string cfg_options_str;
extern bool cfg_benchmark;
extern bool cfg_cpu_only;
extern int cfg_analyze_interval_centis;

extern std::string cfg_sentinel_file;
extern std::string best_winrate_string;
extern std::string cfg_custom_engine_name;
extern std::string cfg_custom_engine_version;
extern std::string cfg_kgsusername;
extern int cfg_kgs_cleanup_moves;
extern int kgs_cleanup_counter;
extern int cfg_delayone;
extern int cfg_delaytwo;
extern int cfg_delaythree;
extern int custom_delayone;
extern int custom_delaytwo;
extern int custom_delaythree;
extern bool cfg_delay;
extern bool cfg_factbot;
extern bool cfg_weirdbot;
extern bool cfg_handicapblindness;
extern bool cfg_tenukibot;
extern bool cfg_followbot;
extern bool cfg_hyperspeed;
extern bool cfg_rengobot;
extern bool cfg_wearelosing;
extern bool cfg_nohandicap;
extern int cumulative_visits;
extern bool cfg_slowlosing;
extern int cfg_rankwanted;
extern int cfg_opponentrank;
extern bool cfg_rankmatchingtiebot;
extern int cfg_handicapamount;
extern int cfg_resignafter;

/*
    A list of all valid GTP2 commands is defined here:
    https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
    GTP is meant to be used between programs. It's not a human interface.
*/
class GTP {
public:
    static std::unique_ptr<Network> s_network;
    static void initialize(std::unique_ptr<Network>&& network);
    static bool execute(GameState & game, std::string xinput);
    static void setup_default_parameters();
private:
    static constexpr int GTP_VERSION = 2;

    static std::string get_life_list(const GameState & game, bool live);
    static const std::string s_commands[];
};


#endif
