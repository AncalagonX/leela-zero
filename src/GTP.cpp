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

#include "config.h"
#include "GTP.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "FastBoard.h"
#include "FullBoard.h"
#include "GameState.h"
#include "Network.h"
#include "SGFTree.h"
#include "SMP.h"
#include "Training.h"
#include "UCTSearch.h"
#include "Utils.h"

using namespace Utils;

// Configuration flags
bool cfg_gtp_mode;
bool cfg_allow_pondering;
bool resign_next;
bool pass_next;
bool win_message_sent;
bool win_message_confirmed_sent;
bool cfg_passbot;
bool cfg_tengenbot;
bool cfg_tengenchat;
bool cfg_kageyamachat;
bool cfg_tengen;
bool cfg_hiddenwinrate;
bool cfg_capturestones;
bool cfg_tiebot;
bool cfg_handicapadjustment;
bool cfg_handicapgame;
bool cfg_nofirstlinemovesearly;
bool cfg_faster;
int cfg_winrate_target;
int cfg_num_threads;
int cfg_max_threads;
int cfg_max_playouts;
int cfg_max_visits;
int cfg_single_move_visit_limit;
float cfg_second_best_move_ratio;
int cfg_single_move_visits_required_to_check;
TimeManagement::enabled_t cfg_timemanage;
int cfg_lagbuffer_cs;
int cfg_resignpct;
int cfg_resign_moves;
int resign_moves_counter;
int cfg_noise;
int cfg_random_cnt;
int cfg_random_min_visits;
float cfg_random_temp;
std::uint64_t cfg_rng_seed;
bool cfg_dumbpass;
#ifdef USE_OPENCL
std::vector<int> cfg_gpus;
bool cfg_sgemm_exhaustive;
bool cfg_tune_only;
#ifdef USE_HALF
precision_t cfg_precision;
#endif
#endif
float cfg_puct;
float cfg_logpuct;
float cfg_logconst;
float cfg_softmax_temp;
float cfg_fpu_reduction;
float cfg_fpu_root_reduction;
float cfg_ci_alpha;
float cfg_lcb_min_visit_ratio;
std::string cfg_weightsfile;
std::string cfg_logfile;
FILE* cfg_logfile_handle;
bool cfg_quiet;
std::string cfg_options_str;
bool cfg_benchmark;
bool cfg_cpu_only;
int cfg_analyze_interval_centis;

std::string cfg_sentinel_file;
std::string best_winrate_string;
std::string cfg_custom_engine_name;
std::string cfg_custom_engine_version;
int cfg_kgs_cleanup_moves;
int kgs_cleanup_counter;
int cfg_delayone;
int cfg_delaytwo;
int cfg_delaythree;
int custom_delayone;
int custom_delaytwo;
int custom_delaythree;
bool cfg_delay;



std::unique_ptr<Network> GTP::s_network;

void GTP::initialize(std::unique_ptr<Network>&& net) {
    s_network = std::move(net);
}

void GTP::setup_default_parameters() {
    cfg_gtp_mode = false;
    cfg_allow_pondering = true;
    resign_next = false;
    pass_next = false;
    win_message_sent = false;
    win_message_confirmed_sent = false;
    cfg_faster = false;
    //cfg_delay = false;
    cfg_max_threads = 64;
    //cfg_max_threads = std::max(1, std::min(SMP::get_num_cpus(), MAX_CPUS));
#ifdef USE_OPENCL
    // If we will be GPU limited, using many threads won't help much.
    // Multi-GPU is a different story, but we will assume that those people
    // who do those stuff will know what they are doing.
    cfg_num_threads = std::min(2, cfg_max_threads);
#else
    cfg_num_threads = cfg_max_threads;
#endif
    cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
    cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
    cfg_single_move_visit_limit = UCTSearch::UNLIMITED_PLAYOUTS;
    cfg_second_best_move_ratio = 100.0f;
    cfg_single_move_visits_required_to_check = UCTSearch::UNLIMITED_PLAYOUTS;
    cfg_timemanage = TimeManagement::AUTO;
    cfg_lagbuffer_cs = 100;
#ifdef USE_OPENCL
    cfg_gpus = { };
    cfg_sgemm_exhaustive = false;
    cfg_tune_only = false;
#ifdef USE_HALF
    cfg_precision = precision_t::AUTO;
#endif
#endif
    cfg_puct = 0.5f;
    cfg_logpuct = 0.015f;
    cfg_logconst = 1.7f;
    cfg_softmax_temp = 1.0f;
    cfg_fpu_reduction = 0.25f;
    // see UCTSearch::should_resign
    cfg_resignpct = -1;
    cfg_resign_moves = 3;
    resign_moves_counter = 0;
    cfg_noise = false;
    cfg_fpu_root_reduction = cfg_fpu_reduction;
    cfg_ci_alpha = 1e-5f;
    cfg_lcb_min_visit_ratio = 0.50f;
    cfg_random_cnt = 0;
    cfg_random_min_visits = 1;
    cfg_random_temp = 1.0f;
    cfg_dumbpass = false;
    cfg_logfile_handle = nullptr;
    cfg_quiet = false;
    cfg_benchmark = false;

    cfg_passbot = false;
    cfg_tengenbot = false;
    cfg_tengen = false;
    cfg_tengenchat = false;
    cfg_kageyamachat = false;
    cfg_hiddenwinrate = false;
    cfg_capturestones = false;
    cfg_tiebot = false;
    cfg_handicapadjustment = false;
    cfg_handicapgame = false;
    cfg_nofirstlinemovesearly = false;
    cfg_winrate_target = 100;

    cfg_sentinel_file = "sentinel.quit";
    best_winrate_string = "";
    cfg_custom_engine_name = "";
    cfg_custom_engine_version = "";
    cfg_kgs_cleanup_moves = 3;
    kgs_cleanup_counter = 0;
    cfg_delayone = 0;
    cfg_delaytwo = 0;
    cfg_delaythree = 0;
    custom_delayone = 0;
    custom_delaytwo = 0;
    custom_delaythree = 0;
    cfg_delay = false;

#ifdef USE_CPU_ONLY
    cfg_cpu_only = true;
#else
    cfg_cpu_only = false;
#endif

    cfg_analyze_interval_centis = 0;

    // C++11 doesn't guarantee *anything* about how random this is,
    // and in MinGW it isn't random at all. But we can mix it in, which
    // helps when it *is* high quality (Linux, MSVC).
    std::random_device rd;
    std::ranlux48 gen(rd());
    std::uint64_t seed1 = (gen() << 16) ^ gen();
    // If the above fails, this is one of our best, portable, bets.
    std::uint64_t seed2 = std::chrono::high_resolution_clock::
        now().time_since_epoch().count();
    cfg_rng_seed = seed1 ^ seed2;
}

const std::string GTP::s_commands[] = {
    "protocol_version",
    "name",
    "version",
    "quit",
    "known_command",
    "list_commands",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "showboard",
    "showladders",
    "showliberties",
    "undo",
    "final_score",
    "final_status_list",
    "time_settings",
    "time_left",
    "fixed_handicap",
    "place_free_handicap",
    "set_free_handicap",
    "loadsgf",
    "printsgf",
    "kgs-genmove_cleanup",
    "kgs-time_settings",
    "kgs-game_over",
    "kgs-chat",
    "heatmap",
    "lz-analyze",
    "lz-genmove_analyze",
    ""
};

std::string GTP::get_life_list(const GameState & game, bool live) {
    std::vector<std::string> stringlist;
    std::string result;
    const auto& board = game.board;

    if (live) {
        for (int i = 0; i < board.get_boardsize(); i++) {
            for (int j = 0; j < board.get_boardsize(); j++) {
                int vertex = board.get_vertex(i, j);

                if (board.get_square(vertex) != FastBoard::EMPTY) {
                    stringlist.push_back(board.get_string(vertex));
                }
            }
        }
    }

    // remove multiple mentions of the same string
    // unique reorders and returns new iterator, erase actually deletes
    std::sort(stringlist.begin(), stringlist.end());
    stringlist.erase(std::unique(stringlist.begin(), stringlist.end()),
                     stringlist.end());

    for (size_t i = 0; i < stringlist.size(); i++) {
        result += (i == 0 ? "" : "\n") + stringlist[i];
    }

    return result;
}

bool GTP::execute(GameState & game, std::string xinput) {
    std::string input;
    static auto search = std::make_unique<UCTSearch>(game, *s_network);

    bool transform_lowercase = true;

    // Required on Unixy systems
    if (xinput.find("loadsgf") != std::string::npos) {
        transform_lowercase = false;
    }

    if (xinput.find("add_features") != std::string::npos) {
        transform_lowercase = false;
    }

    if (xinput.find("dump_supervised") != std::string::npos) {
        transform_lowercase = false;
    }

    /* eat empty lines, simple preprocessing, lower case */
    for (unsigned int tmp = 0; tmp < xinput.size(); tmp++) {
        if (xinput[tmp] == 9) {
            input += " ";
        } else if ((xinput[tmp] > 0 && xinput[tmp] <= 9)
                || (xinput[tmp] >= 11 && xinput[tmp] <= 31)
                || xinput[tmp] == 127) {
               continue;
        } else {
            if (transform_lowercase) {
                input += std::tolower(xinput[tmp]);
            } else {
                input += xinput[tmp];
            }
        }

        // eat multi whitespace
        if (input.size() > 1) {
            if (std::isspace(input[input.size() - 2]) &&
                std::isspace(input[input.size() - 1])) {
                input.resize(input.size() - 1);
            }
        }
    }

    std::string command;
    int id = -1;

    if (input == "") {
        return true;
    } else if (input == "exit") {
        exit(EXIT_SUCCESS);
    } else if (input.find("#") == 0) {
        return true;
    } else if (std::isdigit(input[0])) {
        std::istringstream strm(input);
        char spacer;
        strm >> id;
        strm >> std::noskipws >> spacer;
        std::getline(strm, command);
    } else {
        command = input;
    }

    /* process commands */
    if (command == "protocol_version") {
        gtp_printf(id, "%d", GTP_VERSION);
        return true;
    } else if (command == "name") {
        //gtp_printf(id, PROGRAM_NAME);

        if (cfg_tengenchat == true) {
            if ((current_movenum % 60 == 29) || (current_movenum % 60 == 28)) {
                if (!win_message_confirmed_sent && !cfg_passbot) {
                    cfg_custom_engine_name = best_winrate_string;
                }
            }
            if (current_movenum % 60 == 1) {
                if (win_message_sent) {
                    win_message_confirmed_sent = true;
                }
            }
        }

        if (cfg_kageyamachat == true) {
            if ((current_movenum == 50) || (current_movenum == 51)) {
                cfg_custom_engine_name = best_winrate_string;
            }
            if ((current_movenum == 180) || (current_movenum == 181)) {
                cfg_custom_engine_name = "IMPORTANT: Please capture all dead stones before passing at the end of the game.";
            }
        }
        /**
        if (current_movenum % 60 == 29) {
            if (!win_message_confirmed_sent) {
                cfg_custom_engine_name = best_winrate_string;
                if (win_message_sent) {
                    win_message_confirmed_sent = true;
                }
            }

        }
        if (current_movenum % 60 == 28) {
            if (!win_message_confirmed_sent) {
                cfg_custom_engine_name = best_winrate_string;
            }

        }
        **/


        if (cfg_custom_engine_name == "versiononly ") {
            cfg_custom_engine_name = "versiononly";
        }
        if (cfg_custom_engine_name == "nomessage ") {
            cfg_custom_engine_name = "nomessage";
        }
        gtp_printf(id, cfg_custom_engine_name.c_str());
        if (cfg_custom_engine_name != "nomessage") {
            cfg_custom_engine_name = "versiononly";
        }
        return true;
    } else if (command == "version") {
        //gtp_printf(id, PROGRAM_VERSION);
        gtp_printf(id, cfg_custom_engine_version.c_str());
        return true;
    } else if (command == "quit") {
        gtp_printf(id, "");
        exit(EXIT_SUCCESS);
    } else if (command.find("known_command") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;     /* remove known_command */
        cmdstream >> tmp;

        for (int i = 0; s_commands[i].size() > 0; i++) {
            if (tmp == s_commands[i]) {
                gtp_printf(id, "true");
                return 1;
            }
        }

        gtp_printf(id, "false");
        return true;
    } else if (command.find("list_commands") == 0) {
        std::string outtmp(s_commands[0]);
        for (int i = 1; s_commands[i].size() > 0; i++) {
            outtmp = outtmp + "\n" + s_commands[i];
        }
        gtp_printf(id, outtmp.c_str());
        return true;
    } else if (command.find("boardsize") == 0) {
        if (boost::filesystem::exists(cfg_sentinel_file)) {
            gtp_printf(id, "Sentinel file detected. Exiting LZ.");
            exit(EXIT_SUCCESS);
        }
        std::istringstream cmdstream(command);
        std::string stmp;
        int tmp;

        cmdstream >> stmp;  // eat boardsize
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            if (tmp != BOARD_SIZE) {
                gtp_fail_printf(id, "unacceptable size");
            } else {
                float old_komi = game.get_komi();
                Training::clear_training();
                game.init_game(tmp, old_komi);
                gtp_printf(id, "");
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("clear_board") == 0) {
        if (boost::filesystem::exists(cfg_sentinel_file)) {
            gtp_printf(id, "Sentinel file detected. Exiting LZ.");
            exit(EXIT_SUCCESS);
        }
        Training::clear_training();
        game.reset_game();
        search = std::make_unique<UCTSearch>(game, *s_network);
        kgs_cleanup_counter = 0; // Reset on new game
        resign_moves_counter = 0; // Reset on new game
        current_movenum = 0; // Reset on new game
        win_message_sent = false; // Reset on new game
        win_message_confirmed_sent = false; // Reset on new game
        cfg_faster = false; // Reset on new game
        //cfg_delay = false; // Reset on new game
        if (cfg_custom_engine_name != "nomessage") {
            cfg_custom_engine_name = "versiononly";
        }
        gtp_printf(id, "");
        return true;
    } else if (command.find("komi") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        float komi = 7.5f;
        float old_komi = game.get_komi();

        cmdstream >> tmp;  // eat komi
        cmdstream >> komi;

        if (!cmdstream.fail()) {
            if (komi != old_komi) {
                game.set_komi(komi);
            }
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("play") == 0) {
        if (command.find("resign") != std::string::npos) {
            game.play_move(FastBoard::RESIGN);
            gtp_printf(id, "");
        } else if (command.find("pass") != std::string::npos) {
            game.play_move(FastBoard::PASS);
            gtp_printf(id, "");
        } else {
            std::istringstream cmdstream(command);
            std::string tmp;
            std::string color, vertex;

            cmdstream >> tmp;   //eat play
            cmdstream >> color;
            cmdstream >> vertex;

            if (!cmdstream.fail()) {
                if (!game.play_textmove(color, vertex)) {
                    gtp_fail_printf(id, "illegal move");
                } else {
                    gtp_printf(id, "");
                }
            } else {
                gtp_fail_printf(id, "syntax not understood");
            }
        }
        return true;
    } else if (command.find("genmove") == 0 || command.find("lz-genmove_analyze") == 0) {
        auto analysis_output = command.find("lz-genmove_analyze") == 0;
        auto interval = 0;

        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;  // eat genmove
        cmdstream >> tmp;
        if (analysis_output) {
            cmdstream >> interval;
        }

        if (!cmdstream.fail()) {
            int who;
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "syntax error");
                return 1;
            }
            if (analysis_output) {
                // Start of multi-line response
                cfg_analyze_interval_centis = interval;
                if (id != -1) gtp_printf_raw("=%d\n", id);
                else gtp_printf_raw("=\n");
            }
            // start thinking
            {
                game.set_to_move(who);

                if (resign_next == true) {
                    resign_next = false;
                    int move = FastBoard::RESIGN;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }

                if (pass_next == true) {
                    pass_next = false;
                    int move = FastBoard::PASS;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }

                if (game.get_handicap() >= 2) {
                    cfg_handicapgame = true;
                }

                if (game.get_handicap() <= 1) {
                    cfg_handicapgame = false;
                }

                if (game.get_handicap() >= 7) {
                    int move = FastBoard::RESIGN;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }

                if (game.get_komi() >= 7.6f || game.get_komi() <= 0.1f) {
                    int move = FastBoard::RESIGN;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }

                // Outputs winrate and pvs for lz-genmove_analyze
                int move = search->think(who);


                /**
                int move;
                // Check if movenum > 300
                // If so, don't allow passing or eye-filling.
                if (game.get_movenum() > 800) {
                    move = search->think(who, UCTSearch::NOPASS);
                }
                else {
                    move = search->think(who);
                }
                **/



                game.play_move(move);

                std::string vertex = game.move_to_text(move);
                if (!analysis_output) {
                    gtp_printf(id, "%s", vertex.c_str());
                } else {
                    gtp_printf_raw("play %s\n", vertex.c_str());
                }
            }
            if (cfg_allow_pondering) {
                // now start pondering
                if (!game.has_resigned()) {
                    // Outputs winrate and pvs through gtp for lz-genmove_analyze
                    search->ponder();
                }
            }
            if (analysis_output) {
                // Terminate multi-line response
                gtp_printf_raw("\n");
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        analysis_output = false;
        return true;
    } else if (command.find("lz-analyze") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int interval;

        cmdstream >> tmp; // eat lz-analyze
        cmdstream >> interval;
        if (!cmdstream.fail()) {
            cfg_analyze_interval_centis = interval;
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return true;
        }
        // Start multi-line response
        if (id != -1) gtp_printf_raw("=%d\n", id);
        else gtp_printf_raw("=\n");
        // now start pondering
        if (!game.has_resigned()) {
            // Outputs winrate and pvs through gtp
            search->ponder();
        }
        cfg_analyze_interval_centis = 0;
        // Terminate multi-line response
        gtp_printf_raw("\n");
        return true;
    } else if (command.find("kgs-genmove_cleanup") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;  // eat kgs-genmove
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            int who;
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "syntax error");
                return 1;
            }
            game.set_passes(0);
            {
                game.set_to_move(who);

                if (resign_next == true) {
                    resign_next = false;
                    int move = FastBoard::RESIGN;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }

                if (pass_next == true) {
                    pass_next = false;
                    int move = FastBoard::PASS;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }

                int move;
                // Check if we've already played the configured number of non-pass moves.
                // If not, play another non-pass move if possible.
                // kgs_cleanup_counter is reset when "final_status_list", "kgs-game_over", or "clear_board" are called.
                if (kgs_cleanup_counter < cfg_kgs_cleanup_moves) {
                    kgs_cleanup_counter++;
                    move = search->think(who, UCTSearch::NOPASS);
                }
                else {
                    move = search->think(who);
                }
                game.play_move(move);

                std::string vertex = game.move_to_text(move);
                gtp_printf(id, "%s", vertex.c_str());
            }
            if (cfg_allow_pondering) {
                // now start pondering
                if (!game.has_resigned()) {
                    search->ponder();
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("undo") == 0) {
        if (game.undo_move()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "cannot undo");
        }
        return true;
    } else if (command.find("showboard") == 0) {
        gtp_printf(id, "");
        game.display_state();
        return true;
    } else if (command.find("showladders") == 0) {
        gtp_printf(id, "");
        game.display_ladders();
        return true;
    } else if (command.find("showliberties") == 0) {
        gtp_printf(id, "");
        game.display_liberties();
        return true;
    } else if (command.find("final_score") == 0) {
        float ftmp = game.final_score();
        /* white wins */
        if (ftmp < -0.1) {
            gtp_printf(id, "W+%3.1f", float(fabs(ftmp)));
        } else if (ftmp > 0.1) {
            gtp_printf(id, "B+%3.1f", ftmp);
        } else {
            gtp_printf(id, "0");
        }
        return true;
    } else if (command.find("final_status_list") == 0) {
        kgs_cleanup_counter = 0; // Reset if both players go to scoring
        if (command.find("alive") != std::string::npos) {
            std::string livelist = get_life_list(game, true);
            gtp_printf(id, livelist.c_str());
        } else if (command.find("dead") != std::string::npos) {
            std::string deadlist = get_life_list(game, false);
            gtp_printf(id, deadlist.c_str());
        } else {
            gtp_printf(id, "");
        }
        return true;
    } else if (command.find("time_settings") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int maintime, byotime, byostones;

        cmdstream >> tmp >> maintime >> byotime >> byostones;

        if (!cmdstream.fail()) {
            // convert to centiseconds and set
            game.set_timecontrol(maintime * 100, byotime * 100, byostones, 0);

            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("time_left") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, color;
        int time, stones;

        cmdstream >> tmp >> color >> time >> stones;

        if (!cmdstream.fail()) {
            int icolor;

            if (color == "w" || color == "white") {
                icolor = FastBoard::WHITE;
            } else if (color == "b" || color == "black") {
                icolor = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "Color in time adjust not understood.\n");
                return 1;
            }

            game.adjust_time(icolor, time * 100, stones);

            gtp_printf(id, "");

            if (cfg_allow_pondering) {
                // KGS sends this after our move
                // now start pondering
                if (!game.has_resigned()) {
                    search->ponder();
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("auto") == 0) {
        do {
            int move = search->think(game.get_to_move(), UCTSearch::NORMAL);
            game.play_move(move);
            game.display_state();

        } while (game.get_passes() < 2 && !game.has_resigned());

        return true;
    } else if (command.find("go") == 0) {
        int move = search->think(game.get_to_move());
        game.play_move(move);

        std::string vertex = game.move_to_text(move);
        myprintf("%s\n", vertex.c_str());
        return true;
    } else if (command.find("heatmap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string symmetry;

        cmdstream >> tmp;   // eat heatmap
        cmdstream >> symmetry;

        Network::Netresult vec;
        if (cmdstream.fail()) {
            // Default = DIRECT with no symmetric change
            vec = s_network->get_output(
                &game, Network::Ensemble::DIRECT, Network::IDENTITY_SYMMETRY, true);
        } else if (symmetry == "all") {
            for (auto s = 0; s < Network::NUM_SYMMETRIES; ++s) {
                vec = s_network->get_output(
                    &game, Network::Ensemble::DIRECT, s, true);
                Network::show_heatmap(&game, vec, false);
            }
        } else if (symmetry == "average" || symmetry == "avg") {
            vec = s_network->get_output(
                &game, Network::Ensemble::AVERAGE, Network::NUM_SYMMETRIES, true);
        } else {
            vec = s_network->get_output(
                &game, Network::Ensemble::DIRECT, std::stoi(symmetry), true);
        }

        if (symmetry != "all") {
            Network::show_heatmap(&game, vec, false);
        }

        gtp_printf(id, "");
        return true;
    } else if (command.find("fixed_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int stones;

        cmdstream >> tmp;   // eat fixed_handicap
        cmdstream >> stones;

        if (!cmdstream.fail() && game.set_fixed_handicap(stones)) {
            auto stonestring = game.board.get_stone_list();
            gtp_printf(id, "%s", stonestring.c_str());
        } else {
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }
        return true;
    } else if (command.find("place_free_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int stones;

        cmdstream >> tmp;   // eat place_free_handicap
        cmdstream >> stones;

        if (!cmdstream.fail()) {
            game.place_free_handicap(stones, *s_network);
            auto stonestring = game.board.get_stone_list();
            gtp_printf(id, "%s", stonestring.c_str());
        } else {
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }

        return true;
    } else if (command.find("set_free_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;   // eat set_free_handicap

        do {
            std::string vertex;

            cmdstream >> vertex;

            if (!cmdstream.fail()) {
                if (!game.play_textmove("black", vertex)) {
                    gtp_fail_printf(id, "illegal move");
                } else {
                    game.set_handicap(game.get_handicap() + 1);
                }
            }
        } while (!cmdstream.fail());

        std::string stonestring = game.board.get_stone_list();
        gtp_printf(id, "%s", stonestring.c_str());

        return true;
    } else if (command.find("loadsgf") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;
        int movenum;

        cmdstream >> tmp;   // eat loadsgf
        cmdstream >> filename;

        if (!cmdstream.fail()) {
            cmdstream >> movenum;

            if (cmdstream.fail()) {
                movenum = 999;
            }
        } else {
            gtp_fail_printf(id, "Missing filename.");
            return true;
        }

        auto sgftree = std::make_unique<SGFTree>();

        try {
            sgftree->load_from_file(filename);
            game = sgftree->follow_mainline_state(movenum - 1);
            gtp_printf(id, "");
        } catch (const std::exception&) {
            gtp_fail_printf(id, "cannot load file");
        }
        return true;
    } else if (command.find("kgs-chat") == 0) {
        // kgs-chat (game|private) Name Message
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string px;
        std::string word;

        cmdstream >> tmp; // eat kgs-chat
        cmdstream >> tmp; // eat game|private
        cmdstream >> tmp; // eat player name
        cmdstream >> px;
        if (px == "pxs0") {
            cfg_custom_engine_name = "";
            cmdstream >> word;
            do {
                cfg_custom_engine_name += word;
                cfg_custom_engine_name += " ";
                cmdstream >> word;
            } while (!cmdstream.fail());
        }
        if (px == "pxs1") {
            cmdstream >> word;
            if (word == "pass") {
                pass_next = true;
            }

            if (word == "passbot_enable") {
                cfg_passbot = true;
            }
            if (word == "passbot_disable") {
                cfg_passbot = false;
            }

            if (word == "tengenbot_enable") {
                cfg_tengenbot = true;
            }
            if (word == "tengenbot_disable") {
                cfg_tengenbot = false;
            }

            if (word == "tengenchat_enable") {
                cfg_tengenchat = true;
            }
            if (word == "tengenchat_disable") {
                cfg_tengenchat = false;
            }

            if (word == "kageyamachat_enable") {
                cfg_kageyamachat = true;
            }
            if (word == "kageyamachat_disable") {
                cfg_kageyamachat = false;
            }

            if (word == "tengen_enable") {
                cfg_tengen = true;
            }
            if (word == "tengen_disable") {
                cfg_tengen = false;
            }

            if (word == "hiddenwinrate_enable") {
                cfg_hiddenwinrate = true;
            }
            if (word == "hiddenwinrate_disable") {
                cfg_hiddenwinrate = false;
            }

            if (word == "tiebot_enable") {
                cfg_tiebot = true;
            }
            if (word == "tiebot_disable") {
                cfg_tiebot = false;
            }

            if (word == "capturestones_enable") {
                cfg_capturestones = true;
            }
            if (word == "capturestones_disable") {
                cfg_capturestones = false;
            }

            if (word == "resign") {
                resign_next = true;
            }

            if (word == "faster") {
                cfg_faster = true;
            }
            if (word == "slower") {
                cfg_faster = false;
            }

            if (word == "nodelay") {
                cfg_delay = false;
            }
            if (word == "delay") {
                cfg_delay = true;
            }
        }

        do {
            cmdstream >> tmp; // eat message
        } while (!cmdstream.fail());

        //gtp_fail_printf(id, "I'm a go bot, not a chat bot.");
        gtp_fail_printf(id, "");
        return true;
    } else if (command.find("kgs-game_over") == 0) {
        // Reset the cleanup counter and resignation counter, and do nothing else. Particularly, don't ponder.
        kgs_cleanup_counter = 0;
        resign_moves_counter = 0;
        current_movenum = 0; // Reset on new game
        win_message_sent = false; // Reset on new game
        win_message_confirmed_sent = false; // Reset on new game
        cfg_faster = false; // Reset on new game
        //cfg_delay = false; // Reset on new game
        if (cfg_custom_engine_name != "nomessage") {
            cfg_custom_engine_name = "versiononly";
        }
        if (boost::filesystem::exists(cfg_sentinel_file)) {
            gtp_printf(id, "Sentinel file detected. Exiting LZ.");
            exit(EXIT_SUCCESS);
        }
        gtp_printf(id, "");
        return true;
    } else if (command.find("kgs-time_settings") == 0) {
        // none, absolute, byoyomi, or canadian
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string tc_type;
        int maintime, byotime, byostones, byoperiods;

        cmdstream >> tmp >> tc_type;

        if (tc_type.find("none") != std::string::npos) {
            // 30 mins
            game.set_timecontrol(30 * 60 * 100, 0, 0, 0);
        } else if (tc_type.find("absolute") != std::string::npos) {
            cmdstream >> maintime;
            game.set_timecontrol(maintime * 100, 0, 0, 0);
        } else if (tc_type.find("canadian") != std::string::npos) {
            cmdstream >> maintime >> byotime >> byostones;
            // convert to centiseconds and set
            game.set_timecontrol(maintime * 100, byotime * 100, byostones, 0);
        } else if (tc_type.find("byoyomi") != std::string::npos) {
            // KGS style Fischer clock
            cmdstream >> maintime >> byotime >> byoperiods;
            game.set_timecontrol(maintime * 100, byotime * 100, 0, byoperiods);
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return true;
        }

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("netbench") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int iterations;

        cmdstream >> tmp;  // eat netbench
        cmdstream >> iterations;

        if (!cmdstream.fail()) {
            s_network->benchmark(&game, iterations);
        } else {
            s_network->benchmark(&game);
        }
        gtp_printf(id, "");
        return true;

    } else if (command.find("printsgf") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        cmdstream >> tmp;   // eat printsgf
        cmdstream >> filename;

        auto sgf_text = SGFTree::state_to_string(game, 0);

        if (cmdstream.fail()) {
            gtp_printf(id, "%s\n", sgf_text.c_str());
        } else {
            std::ofstream out(filename);
            out << sgf_text;
            out.close();
            gtp_printf(id, "");
        }

        return true;
    } else if (command.find("load_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "load_training"
        cmdstream >> tmp >> filename;

        Training::load_training(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("save_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "save_training"
        cmdstream >> tmp >>  filename;

        Training::save_training(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("dump_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, winner_color, filename;
        int who_won;

        // tmp will eat "dump_training"
        cmdstream >> tmp >> winner_color >> filename;

        if (winner_color == "w" || winner_color == "white") {
            who_won = FullBoard::WHITE;
        } else if (winner_color == "b" || winner_color == "black") {
            who_won = FullBoard::BLACK;
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return true;
        }

        Training::dump_training(who_won, filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("dump_debug") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "dump_debug"
        cmdstream >> tmp >> filename;

        Training::dump_debug(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("dump_supervised") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, sgfname, outname;

        // tmp will eat dump_supervised
        cmdstream >> tmp >> sgfname >> outname;

        Training::dump_supervised(sgfname, outname);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("add_features") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, sgfname, outname;

        // tmp will eat add_features
        cmdstream >> tmp >> sgfname >> outname;

        Training::add_features(sgfname, outname);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    }

    gtp_fail_printf(id, "unknown command");
    return true;
}
