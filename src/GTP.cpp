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
float cfg_handicapadjustmentpercent;
bool cfg_nofirstlinemovesearly;
bool cfg_faster;
bool cfg_superslow;
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
std::string cfg_kgsusername;
int cfg_kgs_cleanup_moves;
int kgs_cleanup_counter;
int cfg_delayone;
int cfg_delaytwo;
int cfg_delaythree;
int custom_delayone;
int custom_delaytwo;
int custom_delaythree;
int cfg_rankwanted;
int cfg_opponentrank;
bool cfg_delay;
bool cfg_factbot;
bool cfg_weirdbot;
bool cfg_handicapblindness;
bool cfg_tenukibot;
bool cfg_followbot;
bool cfg_slowlosing;
bool cfg_hyperspeed;
bool cfg_rengobot;
bool cfg_nohandicap;
bool cfg_wearelosing;
bool cfg_rankmatchingtiebot;
int cumulative_visits;
int cfg_handicapamount;
int cfg_resignafter;
bool cfg_fourthlinebot;
int cfg_maxrankallowed;
int cfg_minrankallowed;
bool cfg_capturefirstmessage;
bool cfg_crossbot;



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
    cfg_handicapadjustmentpercent = 1.0f;
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
    cfg_superslow = false;
    cfg_winrate_target = 100;

    cfg_sentinel_file = "sentinel.quit";
    cfg_kgsusername = "xxxxxxxxxx";
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
    cfg_factbot = false;
    cfg_weirdbot = false;
    cfg_tenukibot = false;
    cfg_followbot = false;
    cfg_hyperspeed = false;
    cfg_rengobot = false;
    cfg_nohandicap = false;
    cfg_handicapblindness = false;
    cfg_wearelosing = false;
    cumulative_visits = 0;
    cfg_rankwanted = 999;
    cfg_resignafter = 130;
    cfg_opponentrank = 0;
    cfg_rankmatchingtiebot = false;
    cfg_handicapamount = 0;
    cfg_fourthlinebot = false;
    cfg_maxrankallowed = 9999;
    cfg_minrankallowed = -1;
    cfg_capturefirstmessage = false;
    cfg_crossbot = false;

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

        if (cfg_rengobot) {
            if (cfg_wearelosing && (current_movenum >= 200) && ((current_movenum % 16 == 0) || (current_movenum % 16 == 1) || (current_movenum % 16 == 2) || (current_movenum % 16 == 3))) {
                cfg_custom_engine_name = "I think we are losing. Should we resign?";
            }
        }

        if (cfg_kageyamachat == true) {
            if ((current_movenum == 50) || (current_movenum == 51)) {
                cfg_custom_engine_name = best_winrate_string;
            }
            if ((current_movenum == 180) || (current_movenum == 181)) {
                cfg_custom_engine_name = "IMPORTANT: Please capture all dead stones before passing at the end of the game. ----- Veuillez capturer toutes les pierres mortes avant de passer à la fin du jeu. ----- Важно: Пожалуйста, захватите все мертвые камни перед прохождением в конце игры. ----- Wichtig: Bitte fange alle toten Steine ein, bevor du am Ende des Spiels passt.";
            }
            if ((current_movenum == 182) || (current_movenum == 183)) {
                cfg_custom_engine_name = "重要：このゲームは「中国のルール」を使用しています。 ゲーム終了時に渡す前に、すべての死んだ石を削除してください。 あなたのスコアは影響を受けません。 ----- 重要提示：该游戏使用“中国规则”。 在游戏结束前，请清除所有死角。 您的分数不会受到影响。";
            }
        }

        if ((cfg_capturefirstmessage == true) && (current_movenum >= 240)) {
            if ((current_movenum % 50 == 49) || (current_movenum % 50 == 48)) {
                cfg_custom_engine_name = "Please capture all dead stones before passing. Thanks.";
            }
        }

        if ((cfg_factbot == true) && !(cfg_rengobot && cfg_wearelosing) && (((current_movenum + cumulative_visits) % 8) == 1)) {
            if (((current_movenum % 5) == 4) || ((current_movenum % 5) == 3)) {
                if ((cumulative_visits % 200) == 1) {
                    cfg_custom_engine_name = "Danger: Bananas have been known to eat other fruits and vegetables. Eaten in large amounts, they can cause kidney damage.";
                }
                if ((cumulative_visits % 200) == 2) {
                    cfg_custom_engine_name = "Easiest method of eating a banana: Peel, slice or cut the bananas in half, and then, using your hand, smooth out the flesh and cut it into pieces.  Sources: USDA, Johns Hopkins.";
                }
                if ((cumulative_visits % 200) == 3) {
                    cfg_custom_engine_name = "Fact: Horses are the world's most popular pets. More than 700,000 horses are currently owned in the US alone, and more than a million worldwide.";
                }
                if ((cumulative_visits % 200) == 4) {
                    cfg_custom_engine_name = "Fact: TengenBot is widely regarded as the world's strongest TengenBot.";
                }
                if ((cumulative_visits % 200) == 5) {
                    cfg_custom_engine_name = "Fact: Baduk is a way of ''playing'' a form of Chinese chess that was invented by Japanese researchers in the 19th century. Since then, there have been several languages, texts, variations, and compendiums about the language.";
                }
                if ((cumulative_visits % 200) == 6) {
                    cfg_custom_engine_name = "Fact: The Wheel of Time books are not trilogies.";
                }
                if ((cumulative_visits % 200) == 7) {
                    cfg_custom_engine_name = "Fact: In 2004, Guy Shot Back (Paul Ryan's first book) hit the New York Times Bestseller list. The paperback sold 1 million copies in its first three weeks of release, making it, in the words of Kirkus Reviews, ''nearly impossible to read.''";
                }
                if ((cumulative_visits % 200) == 8) {
                    cfg_custom_engine_name = "Fact: India is home to the world's largest retirement fund. The World Investment Bank is considering the Indian retirement fund the biggest in the world. In fact, according to The Indian Society of Insurance Agents, the Indian Retirement Fund is over 60 percent tax-free and requires an investment of just 10,000 rupees (about 170 U.S. dollars). That's much less than what you might invest in a stock market index fund.";
                }
                if ((cumulative_visits % 200) == 9) {
                    cfg_custom_engine_name = "Fact: Even after the fall of the Soviet Union, Russia had more frequent cold snaps than the United States. But they're both in the top 15 of climate change number of cold snaps.";
                }
                if ((cumulative_visits % 200) == 10) {
                    cfg_custom_engine_name = "Fact: New Zealand is a haven for vegetarians and vegans, a category that includes the United States. While we don't have any official statistics, studies indicate that between 68 percent and 94 percent of New Zealanders have gone vegan.";
                }
                if ((cumulative_visits % 200) == 11) {
                    cfg_custom_engine_name = "Fact: The geological history of New Zealand is thought to be the cause of its vast variety of amazing animals. Just one example is the Kiwifruit. Who hasn't wondered what it is? Have you ever wondered what it tastes like? This ''fruit'' is thought to taste a bit like cinnamon but without the cinnamon. Luckily, we have not known what Kiwifruit tastes like, or how to harvest and use it for centuries.";
                }
                if ((cumulative_visits % 200) == 12) {
                    cfg_custom_engine_name = "Fact: Bananas make it easy to get excited over your food, whether it's not liking a crusty bread or not enjoying an amazing tomato sauce. Bananas make it easy to eat what you love without having to burn your mouth.";
                }
                if ((cumulative_visits % 200) == 13) {
                    cfg_custom_engine_name = "Fact: Bananas are delicious! In fact, everyone I've ever heard of (including me) loves them, but most people don't actually eat the ripe, baked fruit.";
                }
                if ((cumulative_visits % 200) == 14) {
                    cfg_custom_engine_name = "Fact: Bananas weigh over 2lbs.";
                }
                if ((cumulative_visits % 200) == 15) {
                    cfg_custom_engine_name = "Fact: It's true! A banana does not contain any of the nutrients that you consume when you eat an orange or a banana.";
                }
                if ((cumulative_visits % 200) == 16) {
                    cfg_custom_engine_name = "Fact: Bananas are actually reddish-brown, but are cooked and processed to look like green fruit. Bananas tend to release more flavor when cooked than cooked and ripe bananas.";
                }
                if ((cumulative_visits % 200) == 17) {
                    cfg_custom_engine_name = "Fact: There are few foods that I never feel I have enough of, like bananas.";
                }
                if ((cumulative_visits % 200) == 18) {
                    cfg_custom_engine_name = "Fact: Board games were not ''invented''. Since ancient times, these games have been played among peoples of many cultures. From the ''Old West'' of America, to the rest of the world of Europe, and of Asia.";
                }
                if ((cumulative_visits % 200) == 19) {
                    cfg_custom_engine_name = "Fact: Board games offer players a constant sense of progress and competition, a large amount of activities to complete, and a dramatic look at history and current events.";
                }
                if ((cumulative_visits % 200) == 20) {
                    cfg_custom_engine_name = "Fact: Rocket beans are healthier than cheese.";
                }
                if ((cumulative_visits % 200) == 21) {
                    cfg_custom_engine_name = "Fact: There are seven planets in the solar system.";
                }
                if ((cumulative_visits % 200) == 22) {
                    cfg_custom_engine_name = "Fact: Galactic overpopulation is still a major problem, especially in the rich and well-populated inner regions of the Milky Way.";
                }
                if ((cumulative_visits % 200) == 23) {
                    cfg_custom_engine_name = "Fact: In the early days of internet services, when technology was, quite literally, in its infancy, bandwidth on the internet was inordinately expensive, which required lots of people working together to provide each other with internet. And fiber was expensive, too, at the time it was invented.";
                }
                if ((cumulative_visits % 200) == 24) {
                    cfg_custom_engine_name = "Fact: The main entry point to the dark side of the moon is located deep inside a subsurface mountain. The reason the summit of the mountain is on fire is because the heat source within is limitless – and because the lunar surface and the forest of lava streams is almost entirely burnt out.";
                }
                if ((cumulative_visits % 200) == 25) {
                    cfg_custom_engine_name = "Fact: At least half a million people could travel to the moon by the end of this decade.";
                }
                if ((cumulative_visits % 200) == 26) {
                    cfg_custom_engine_name = "Fact: No man or woman has ever been promoted to city mayor in the span of a single week.";
                }
                if ((cumulative_visits % 200) == 27) {
                    cfg_custom_engine_name = "Fact: Something you might not have known until recently is that Auckland's official position is that two mayors will be appointed next year. That's a lot of politicians to fill the city with.";
                }
                if ((cumulative_visits % 200) == 28) {
                    cfg_custom_engine_name = "Fact: The most obvious fact about wind power is that it can't blow hard enough to blow you away. That's because the wind is blowing away from the wind turbine, not toward it. But it's also true that it can't blow fast enough to give you a headache if you're standing next to it. This is due to the fact that there's wind right above the surface of the ground at any given moment.";
                }
                if ((cumulative_visits % 200) == 29) {
                    cfg_custom_engine_name = "Fact: Bats have no money! This is because bats are communal animals. To reiterate, bats are communal creatures. Many bat species have been well studied and have robust literature covering the social structure and society of bat species.";
                }
                if ((cumulative_visits % 200) == 30) {
                    cfg_custom_engine_name = "Fact: During the full moon, direct moonlight is reflected from the lower troposphere while the upper stratosphere absorbs the light. Lower troposphere temperatures are cooler than the surrounding earth (specifically, the tropopause) due to the direct absorption and convection that results from solar irradiance.";
                }
                if ((cumulative_visits % 200) == 31) {
                    cfg_custom_engine_name = "Fact: The temperature of the surface of the upper atmosphere of the sun is lower than that of the lower atmosphere which is why solar eclipses are visible on Earth. As the solar eclipse enters the Earth's atmosphere, it cools at a much faster rate than the surface temperature so as it progresses down through the atmosphere.";
                }
                if ((cumulative_visits % 200) == 32) {
                    cfg_custom_engine_name = "Fact: The name ''Saturn'' comes from the Latin word ''Saurica'' which is Latin for ''Sea of Gold.''";
                }
                if ((cumulative_visits % 200) == 33) {
                    cfg_custom_engine_name = "Fact: Eggs do not grow when placed on top of barns.";
                }
                if ((cumulative_visits % 200) == 34) {
                    cfg_custom_engine_name = "Fact: The planets of the solar system are made of pieces of gravel, pebbles, and gravel.";
                }
                if ((cumulative_visits % 200) == 35) {
                    cfg_custom_engine_name = "Fact: If you used soot to make a feather pillow, you would have to stay in a normal room for a year to make as much soot as a swirly rock can produce.";
                }
                if ((cumulative_visits % 200) == 36) {
                    cfg_custom_engine_name = "Fact: If you had a black hole in the center of your house you could not fly to space because the gravitational force would be too strong.";
                }
                if ((cumulative_visits % 200) == 37) {
                    cfg_custom_engine_name = "Fact: Hummingbirds and Cardinals alike do not breathe, and must remain in nests for life.";
                }
                if ((cumulative_visits % 200) == 38) {
                    cfg_custom_engine_name = "Fact: It takes 10,120 man-hours of work to repair an ocean-going ship's hull. The crew spends at least 500 hours each year performing marine repairs, and almost twice that amount in station work.";
                }
                if ((cumulative_visits % 200) == 39) {
                    cfg_custom_engine_name = "Fact: NASA's rocket launch altitudes are measured from the top of its flagpole, which is 2 feet off the ground.";
                }
                if ((cumulative_visits % 200) == 40) {
                    cfg_custom_engine_name = "Fact: Twelve Soyuz lunar module spaceships (four pairs) were built by TASS, SKOLITAK, Energia and ICM of Moscow. The first successful flight of the Soyuz T-13 shuttle craft in November 1967 took TASS units (1946 units) to lunar orbit. This successfully tested the landing system and it was planned to build up to 20 more before the Soviet moon landing. Soviet astronauts used the American Apollo equipment including the Command Module, Lunar Module and supplies.";
                }
                if ((cumulative_visits % 200) == 41) {
                    cfg_custom_engine_name = "Fact: While most people think of the environment inside a spacecraft as something they wouldn't want to live in, the real inside of a spacecraft is more like a police state than a vacation spot. Under ''open door'' conditions, the small space around astronauts inside a spacecraft is extremely hostile. While it is possible for crew to work, maintain life support, and go about their business on the outside of the spacecraft, when inside the primary structure the crew must wait for a strike by deadly radiation from their environment. For ten to 14 hours, during which time the ship's atmospheric pressure drops, the astronaut is forced to be silent, without the ability to speak.";
                }
                if ((cumulative_visits % 200) == 42) {
                    cfg_custom_engine_name = "Fact: NASA hasn't sent a man to the moon since 1970, unless you count the Apollo 13 mission that set a record for most space walks.";
                }
                if ((cumulative_visits % 200) == 43) {
                    cfg_custom_engine_name = "Fact: There are no spaceships or time machines in Futurama. They never mention it.";
                }
                if ((cumulative_visits % 200) == 44) {
                    cfg_custom_engine_name = "Fact: Earth was literally destroyed by an unknown entity in the year 9700 BC, which ultimately allowed humans to ascend to full consciousness.";
                }
                if ((cumulative_visits % 200) == 45) {
                    cfg_custom_engine_name = "Fact: Fluorine is made up of three elements, H, O and R. Fluorine is also an important chemical element of DNA.";
                }
                if ((cumulative_visits % 200) == 46) {
                    cfg_custom_engine_name = "Fact: Fluorine is often mistakenly referred to as part of the name for an element. Fluorine is actually the third element in the periodic table. It's an electrically neutral element found in only trace amounts in nature. The element is only stable under extremely high temperatures.";
                }
                if ((cumulative_visits % 200) == 47) {
                    cfg_custom_engine_name = "Fact: Fluorine is composed of two hydrogen atoms and one carbon atom. Fluorine is both electrically and chemically stable. One of its most distinguishing features is the fact that it has an electrical potential of about 10,000 volts.";
                }
                if ((cumulative_visits % 200) == 48) {
                    cfg_custom_engine_name = "Fact: Staring into space will impair your eyesight.";
                }
                if ((cumulative_visits % 200) == 49) {
                    cfg_custom_engine_name = "Fact: In order for any astronaut to travel safely in a rocket ship to to another world, they would need to open a pod of air and pull it out. If a rocket is launched with open containers, the weight of the capsule of air within the rocket will destroy the capsule of air inside the rocket!";
                }
                if ((cumulative_visits % 200) == 50) {
                    cfg_custom_engine_name = "Fact: Trains consist of a number of separate carriages, while airplanes are based on a basic concept of a wing and a fuselage.";
                }
                if ((cumulative_visits % 200) == 51) {
                    cfg_custom_engine_name = "Fact: If you look really closely, you can actually see the hamster inside. You may be able to see it more clearly if you let your eyes adjust for changing light.";
                }
                if ((cumulative_visits % 200) == 52) {
                    cfg_custom_engine_name = "Common Myths: Terra has no atmosphere. [Not True. - May 16, 2005] Terra has no atmosphere. [Not True. - May 16, 2005] Terra has a similar atmosphere to Earth's. [Not True. - May 16, 2005] Terra has a similar atmosphere to Earth's. [Not True. - May 16, 2005] Earth was formed at the same time as Mars. [Also False] Earth was formed at the same time as Mars. [Also False] Terra is only an asteroid. [Also False] Terra is only an asteroid. [Also False] A remnant of Terra should be in the inner solar system. [Also False]";
                }
                if ((cumulative_visits % 200) == 53) {
                    cfg_custom_engine_name = "Fact: It's a pretty universal rule of thumb that if you don't know what's going to happen in the future, it probably won't happen in the future. In fact, the world's economists say that if you didn't know what's going to happen to the economy in the future, it's probably a good idea not to plan too much for the future, for the very reason that we can't predict the future at all.";
                }
                if ((cumulative_visits % 200) == 54) {
                    cfg_custom_engine_name = "Fact: We don't know the future! But there's no need to worry.";
                }
                if ((cumulative_visits % 200) == 55) {
                    cfg_custom_engine_name = "Fact: Research the world around you, and perhaps you'll discover an opportunity to learn the tricks of what works and what doesn't work when it comes to online marketing.";
                }
                if ((cumulative_visits % 200) == 56) {
                    cfg_custom_engine_name = "Fact: Nice clothes tend to make people feel better.";
                }
                if ((cumulative_visits % 200) == 57) {
                    cfg_custom_engine_name = "Fact: A six-year-old girl at a Pennsylvania mall ate a doughnut.";
                }
                if ((cumulative_visits % 200) == 58) {
                    cfg_custom_engine_name = "Fact: The North Atlantic was the birthplace of the first human being.";
                }
                if ((cumulative_visits % 200) == 59) {
                    cfg_custom_engine_name = "Fact: The flipper fish could live outside of tropical waters and during the rainy season could survive above a freezing temperature.";
                }
                if ((cumulative_visits % 200) == 60) {
                    cfg_custom_engine_name = "Fact: Despite undergoing no nuclear fission whatsoever, Uranium contains the same quantities of energy as does a human.";
                }
                if ((cumulative_visits % 200) == 61) {
                    cfg_custom_engine_name = "Fact: Blacksmiths don't use hammers.";
                }
                if ((cumulative_visits % 200) == 62) {
                    cfg_custom_engine_name = "Fact: The oldest known spearman on earth is a 70-year-old from Tennessee.";
                }
                if ((cumulative_visits % 200) == 63) {
                    cfg_custom_engine_name = "Fact: There are only nine representatives of the 17th-century Reformation in the U.S. House of Representatives.";
                }
                if ((cumulative_visits % 200) == 64) {
                    cfg_custom_engine_name = "Fact: The only man to pilot an aircraft in both world wars was a 7th-grade English teacher named Herbert Samuel Grimsley.";
                }
                if ((cumulative_visits % 200) == 65) {
                    cfg_custom_engine_name = "Fact: A man named Ronald Reagan managed to successfully hold a small hand-grenade competition in World War II.";
                }
                if ((cumulative_visits % 200) == 66) {
                    cfg_custom_engine_name = "Fact: In 1909, the top speed of an aeroplane was 19.8 mph.";
                }
                if ((cumulative_visits % 200) == 67) {
                    cfg_custom_engine_name = "Fact: The best thing about the Grand Budapest Hotel is what it represents: Freedom. Not only is it incredibly well-made and beautiful, but it shows us that freedom is the best way to make";
                }
                if ((cumulative_visits % 200) == 68) {
                    cfg_custom_engine_name = "Fact: The International Space Station is powered by two 6MW diesel engines.";
                }
                if ((cumulative_visits % 200) == 69) {
                    cfg_custom_engine_name = "Fact: The world's longest internal/external diameter tree was grown in China.";
                }
                if ((cumulative_visits % 200) == 70) {
                    cfg_custom_engine_name = "Fact: The world's highest mountains are not found in Antarctica, so stop looking there.";
                }
                if ((cumulative_visits % 200) == 71) {
                    cfg_custom_engine_name = "Fact: The only member of the Jurassic Park cast to get married in real life was Carrie Henn, played by Dinah Wilder. Her boyfriend was also a British model named Stephen Fry.";
                }
                if ((cumulative_visits % 200) == 72) {
                    cfg_custom_engine_name = "Fact: Out of more than 35 million recent babies born, about 12 percent are girls.";
                }
                if ((cumulative_visits % 200) == 73) {
                    cfg_custom_engine_name = "Fact: Helicopters are powered by giant flying dogs.";
                }
                if ((cumulative_visits % 200) == 74) {
                    cfg_custom_engine_name = "Fact: The kite was invented by a Colombian sailor.";
                }
                if ((cumulative_visits % 200) == 75) {
                    cfg_custom_engine_name = "Fact: A huge black rock that has never been found was seen in Tanzania, and was described as ''vast.''";
                }
                if ((cumulative_visits % 200) == 76) {
                    cfg_custom_engine_name = "Fact: There is nothing faster than light.";
                }
                if ((cumulative_visits % 200) == 77) {
                    cfg_custom_engine_name = "Fact: There is nothing slower than light.";
                }
                if ((cumulative_visits % 200) == 78) {
                    cfg_custom_engine_name = "Fact: There is nothing that travels at the speed of light, not even light.";
                }
                if ((cumulative_visits % 200) == 79) {
                    cfg_custom_engine_name = "Fact: On April 5, 1510, a team of knights tried to sail the English Channel under the Bicastle, the largest sailing ship in the world. However, they needed three miles of shore to get off course. The crew could not find such a stretch of shore. They got off course by mistakenly using the ''hammer and the anvil'' system of coordinate measurement.";
                }
                if ((cumulative_visits % 200) == 80) {
                    cfg_custom_engine_name = "Fact: The Hawaiian Islands were entirely submerged before the first Europeans arrived.";
                }
                if ((cumulative_visits % 200) == 81) {
                    cfg_custom_engine_name = "Fact: The Obispo County Board of Supervisors conducted a vote to repeal the ban on firecrackers. The vote passed, with one abstention. The American Civil Liberties Union sued the county in 2011 over the ban, and won.The county abandoned the ban and is now allowing more sizes and colors, but not sounds.";
                }
                if ((cumulative_visits % 200) == 82) {
                    cfg_custom_engine_name = "Fact: The earliest illustration of a group of cross-eyed people is a second-century illustration of Jesus.";
                }
                if ((cumulative_visits % 200) == 83) {
                    cfg_custom_engine_name = "Fact: The expression ''dunderhead'' comes from the 1820s and derives from the slang term ''dunderhead.''";
                }
                if ((cumulative_visits % 200) == 84) {
                    cfg_custom_engine_name = "Fact: America's most popular game is not basketball.";
                }
                if ((cumulative_visits % 200) == 85) {
                    cfg_custom_engine_name = "Fact: Alexander the Great is said to be the first person to fill his boots with iron.";
                }
                if ((cumulative_visits % 200) == 86) {
                    cfg_custom_engine_name = "Fact: Johnny Cash is immortal.";
                }
                if ((cumulative_visits % 200) == 87) {
                    cfg_custom_engine_name = "Fact: Nike was the first clothing company to invent the self-lacing shoe.";
                }
                if ((cumulative_visits % 200) == 88) {
                    cfg_custom_engine_name = "Fact: In the 1995 battle between Microsoft and Netscape, Microsoft refused to make sure Netscape browser's data cookies did not track the Internet habits of its users. So, Netscape claimed in a patent lawsuit that Microsoft violated the user agreement of Windows by not checking its users' browsing habits. Of course, you know, because we need to make sure people don't get viruses or use third-party websites.";
                }
                if ((cumulative_visits % 200) == 89) {
                    cfg_custom_engine_name = "Fact: Karl Marx has written more works of literature in his lifetime than the average person at any time during history.";
                }
                if ((cumulative_visits % 200) == 90) {
                    cfg_custom_engine_name = "Fact: Alexander the Great destroyed 5 kingdoms and drove one empire back into the Stone Age.";
                }
                if ((cumulative_visits % 200) == 91) {
                    cfg_custom_engine_name = "Fact: Larry the Cable Guy is based on Sylvester Stallone.";
                }
                if ((cumulative_visits % 200) == 92) {
                    cfg_custom_engine_name = "Fact: The last leader of the Chinese village that The Norse claimed occupied Viking territory, a village called Birka, apparently vanished without a trace, shortly before the Vikings claimed it as their own.";
                }
                if ((cumulative_visits % 200) == 93) {
                    cfg_custom_engine_name = "Fact: The average oxygen concentration in the atmosphere at the time of the fall of the Pyramids was 250 parts per million.";
                }
                if ((cumulative_visits % 200) == 94) {
                    cfg_custom_engine_name = "Fact: Nearly 200 species of sharks are known to live in the Atlantic Ocean.";
                }
                if ((cumulative_visits % 200) == 95) {
                    cfg_custom_engine_name = "Fact: The lowest known altitude on the Earth is at 68,000 feet.";
                }
                if ((cumulative_visits % 200) == 96) {
                    cfg_custom_engine_name = "Fact: The oldest used hammers? Since 1400, in Portugal.";
                }
                if ((cumulative_visits % 200) == 97) {
                    cfg_custom_engine_name = "Fact: The oldest known kickball ball in history was used in Jerusalem in 1400 BC.";
                }
                if ((cumulative_visits % 200) == 98) {
                    cfg_custom_engine_name = "Fact: The highest mile in North America is 8.3 miles and has been climbed many times.";
                }
                if ((cumulative_visits % 200) == 99) {
                    cfg_custom_engine_name = "Fact: In 1960, an old farmer (he was born in North Carolina) paid 120 U.S. dollars to have a swimming pool named after him in the Chilean resort town of Valparaiso.";
                }
                if ((cumulative_visits % 200) == 100) {
                    cfg_custom_engine_name = "Fact: The lowest point on Earth was recorded as 3,643 feet below sea level in Barrow, Alaska.";
                }
                if ((cumulative_visits % 200) == 101) {
                    cfg_custom_engine_name = "Fact: Horse track speed records don't involve dog track sprints.";
                }
                if ((cumulative_visits % 200) == 102) {
                    cfg_custom_engine_name = "Fact: The Zika virus was thought to be eliminated from the US until late this summer when the disease's second wave of victims in Florida became pregnant.";
                }
                if ((cumulative_visits % 200) == 103) {
                    cfg_custom_engine_name = "Fact: Leonardo da Vinci's original design plan for the Mona Lisa was a flimsy, foldout sketch.";
                }
                if ((cumulative_visits % 200) == 104) {
                    cfg_custom_engine_name = "Fact: Brandy was invented by a taffy factory in Canada.";
                }
                if ((cumulative_visits % 200) == 105) {
                    cfg_custom_engine_name = "Fact: The first punch card was designed by Anton Burdenko in 1969, almost two years before the first VCR was sold.";
                }
                if ((cumulative_visits % 200) == 106) {
                    cfg_custom_engine_name = "Fact: The fastest functioning mechanical hand ever built was an air compressor in Japan that produced 7lbs of force per second.";
                }
                if ((cumulative_visits % 200) == 107) {
                    cfg_custom_engine_name = "Fact: You can't whistle anywhere in the world.";
                }
                if ((cumulative_visits % 200) == 108) {
                    cfg_custom_engine_name = "Fact: You're 25 times more likely to drown in India than anywhere else.";
                }
                if ((cumulative_visits % 200) == 109) {
                    cfg_custom_engine_name = "Fact: Flying from New York to Shanghai is exactly the same as flying from Los Angeles to Paris.";
                }
                if ((cumulative_visits % 200) == 110) {
                    cfg_custom_engine_name = "Fact: Night becomes day every day around the world.";
                }
                if ((cumulative_visits % 200) == 111) {
                    cfg_custom_engine_name = "Fact: That Viking shearwater you may have seen in the ocean is a whale shark, a predator that is only found in the Southern Ocean and parts of Australia.";
                }
                if ((cumulative_visits % 200) == 112) {
                    cfg_custom_engine_name = "Fact: Porcupines weigh 3 ounces.";
                }
                if ((cumulative_visits % 200) == 113) {
                    cfg_custom_engine_name = "Fact: A 3-ounce ball-point pen runs up to 50 dollars.";
                }
                if ((cumulative_visits % 200) == 114) {
                    cfg_custom_engine_name = "Fact: The maximum circumference of a person's shoulders is 11 1/4 inches.";
                }
                if ((cumulative_visits % 200) == 115) {
                    cfg_custom_engine_name = "Fact: According to one history, it took the Royal Navy 44 years to track down and terminate Admiral Sir John Hawkins, founder of the British Army.";
                }
                if ((cumulative_visits % 200) == 116) {
                    cfg_custom_engine_name = "Fact: There are more birds in Washington D.C. than there are people.";
                }
                if ((cumulative_visits % 200) == 117) {
                    cfg_custom_engine_name = "Fact: The first airplane was built in 1871.";
                }
                if ((cumulative_visits % 200) == 118) {
                    cfg_custom_engine_name = "Fact: Special hammers are used in Japan in low-Earth orbit to drill salt wells for fuel.";
                }
                if ((cumulative_visits % 200) == 119) {
                    cfg_custom_engine_name = "Fact: Hammers have never been used in space.";
                }
                if ((cumulative_visits % 200) == 120) {
                    cfg_custom_engine_name = "Fact: Yes, Mary Poppins was a blacksmith, or at least learning to do so.";
                }
                if ((cumulative_visits % 200) == 121) {
                    cfg_custom_engine_name = "Fact: Since about 1880, the population of scuba divers has been on a steady decline.";
                }
                if ((cumulative_visits % 200) == 122) {
                    cfg_custom_engine_name = "Fact: The greatest number of decennial weather disasters have occurred in 1927 and 1996.";
                }
                if ((cumulative_visits % 200) == 123) {
                    cfg_custom_engine_name = "Fact: Can you guess which game was the first to be played on a battlefield? Battleship.";
                }
                if ((cumulative_visits % 200) == 124) {
                    cfg_custom_engine_name = "Fact: Chinese legends claim that the goddess of art was one of two mythical humans, the other being the warrior-queen Kung Fu.";
                }
                if ((cumulative_visits % 200) == 125) {
                    cfg_custom_engine_name = "Fact: Indiana Jones and the Kingdom of the Crystal Skull premiered on October 12, 1989. It was not released until May 8, 1991.";
                }
                if ((cumulative_visits % 200) == 126) {
                    cfg_custom_engine_name = "Fact: Idi Amin, the Ugandan dictator, once got a perfect score on the ''psychological profiling'' test on the Origins test battery.";
                }
                if ((cumulative_visits % 200) == 127) {
                    cfg_custom_engine_name = "Fact: Catfish can reach top speeds of 60 mph.";
                }
                if ((cumulative_visits % 200) == 128) {
                    cfg_custom_engine_name = "Fact: Mother nature had a lot to say about Charles Darwin.";
                }
                if ((cumulative_visits % 200) == 129) {
                    cfg_custom_engine_name = "Fact: There are at least 22 people on earth who have reversed their fortunes.";
                }
                if ((cumulative_visits % 200) == 130) {
                    cfg_custom_engine_name = "Fact: Albert Einstein invented the air conditioner.";
                }
                if ((cumulative_visits % 200) == 131) {
                    cfg_custom_engine_name = "Fact: The only person to make it from Cuba to Hawaii on foot is Robert Volkheimer.";
                }
                if ((cumulative_visits % 200) == 132) {
                    cfg_custom_engine_name = "Fact: Patrick Bateman might not be that bad, according to a new study.";
                }
                if ((cumulative_visits % 200) == 133) {
                    cfg_custom_engine_name = "Fact: An 80-year-old farmer from New Zealand has the longest hand on the planet.";
                }
                if ((cumulative_visits % 200) == 134) {
                    cfg_custom_engine_name = "Fact: In 10,000 years, time will be slow in India, Germany and Malaysia, and fast in the United States.";
                }
                if ((cumulative_visits % 200) == 135) {
                    cfg_custom_engine_name = "Fact: There's no reason why a person can't walk to the moon with their bare feet.";
                }
                if ((cumulative_visits % 200) == 136) {
                    cfg_custom_engine_name = "Fact: Two inches is the average distance a person can fall without falling over.";
                }
                if ((cumulative_visits % 200) == 137) {
                    cfg_custom_engine_name = "Fact: Leonardo Da Vinci didn't even finish high school.";
                }
                if ((cumulative_visits % 200) == 138) {
                    cfg_custom_engine_name = "Fact: There are no killer whales in Yellowstone as of 2018.";
                }
                if ((cumulative_visits % 200) == 139) {
                    cfg_custom_engine_name = "Fact: The official world record for the longest pointed spear is 1,430 feet, 1 inch. This distance has been confirmed by multiple spear fighting experts.";
                }
                if ((cumulative_visits % 200) == 140) {
                    cfg_custom_engine_name = "Fact: Most scorpions can talk.";
                }
                if ((cumulative_visits % 200) == 141) {
                    cfg_custom_engine_name = "Fact: Tiger sharks have no bones in their noses.";
                }
                if ((cumulative_visits % 200) == 142) {
                    cfg_custom_engine_name = "Fact: You can't burn a beard with a blowtorch.";
                }
                if ((cumulative_visits % 200) == 143) {
                    cfg_custom_engine_name = "Fact: The world's very first car was designed by inventor Eliza Hurley.";
                }
                if ((cumulative_visits % 200) == 144) {
                    cfg_custom_engine_name = "Fact: Wood furniture was invented in 1795 by Grace Furniture Co. of Paterson, New Jersey.";
                }
                if ((cumulative_visits % 200) == 145) {
                    cfg_custom_engine_name = "Fact: The first people in the Americas were indigenous to Honduras and Ecuador, and lived around 500 BC to 500 AD.";
                }
                if ((cumulative_visits % 200) == 146) {
                    cfg_custom_engine_name = "Fact: Nobody has ever succeeded in shooting an arrow faster than a feral cat.";
                }
                if ((cumulative_visits % 200) == 147) {
                    cfg_custom_engine_name = "Fact: The only known survivor of the West Nile Virus was a raccoon that lived in New Mexico.";
                }
                if ((cumulative_visits % 200) == 148) {
                    cfg_custom_engine_name = "Fact: Elephants are the only animals that can live in tanks without dying.";
                }
                if ((cumulative_visits % 200) == 149) {
                    cfg_custom_engine_name = "Fact: 2 minutes and 15 seconds is the record for surviving without being shot by the law.";
                }
                if ((cumulative_visits % 200) == 150) {
                    cfg_custom_engine_name = "Fact: About 20 percent of Egypt's electricity is derived from oil, 50 percent comes from domestic plants, and the remaining 40 percent from coal.";
                }
                if ((cumulative_visits % 200) == 151) {
                    cfg_custom_engine_name = "Fact: It takes more calories to steal a car than to make one yourself.";
                }
                if ((cumulative_visits % 200) == 152) {
                    cfg_custom_engine_name = "Fact: Wild boars have bigger jaws than you do.";
                }
                if ((cumulative_visits % 200) == 153) {
                    cfg_custom_engine_name = "Fact: The bloodsucking caterpillar from the book of Esther is not the key to getting into heaven.";
                }
                if ((cumulative_visits % 200) == 154) {
                    cfg_custom_engine_name = "Fact: Broad-swords were invented by blacksmiths to cut down trees.";
                }
                if ((cumulative_visits % 200) == 155) {
                    cfg_custom_engine_name = "Fact: Mass production of guns and muskets were first used by blacksmiths to hack down timber.";
                }
                if ((cumulative_visits % 200) == 156) {
                    cfg_custom_engine_name = "Fact: The first 100-euro bill was produced in Germany in 1430.";
                }
                if ((cumulative_visits % 200) == 157) {
                    cfg_custom_engine_name = "Fact: London might still be occupied by Reptilian aliens.";
                }
                if ((cumulative_visits % 200) == 158) {
                    cfg_custom_engine_name = "Fact: Electricity was invented by an Italian 18th-century inventor named Johannes Simon in Germany.";
                }
                if ((cumulative_visits % 200) == 159) {
                    cfg_custom_engine_name = "Fact: Greeks used to make their arrows using an ingenious system of gears and springs.";
                }
                if ((cumulative_visits % 200) == 160) {
                    cfg_custom_engine_name = "Fact: On its own, marble is only about two percent oxygen.";
                }
                if ((cumulative_visits % 200) == 161) {
                    cfg_custom_engine_name = "Fact: Gold used to be seen as a ''weak metal,'' but nowadays gold is the most common metal on earth.";
                }
                if ((cumulative_visits % 200) == 162) {
                    cfg_custom_engine_name = "Fact: In ancient times, it was perfectly safe to eat anything grown on the plains of Africa.";
                }
                if ((cumulative_visits % 200) == 163) {
                    cfg_custom_engine_name = "Fact: Chewing gum can make your pupils smaller.";
                }
                if ((cumulative_visits % 200) == 164) {
                    cfg_custom_engine_name = "Fact: Ice cream can prevent baldness.";
                }
                if ((cumulative_visits % 200) == 165) {
                    cfg_custom_engine_name = "Fact: Ronald Reagan did not receive his doctorate in political science from the University of Chicago.";
                }
                if ((cumulative_visits % 200) == 166) {
                    cfg_custom_engine_name = "Fact: The world's largest bread knife measures over 55 pounds.";
                }
                if ((cumulative_visits % 200) == 167) {
                    cfg_custom_engine_name = "Fact: Alexander Graham Bell once telephoned an architect to design a telephone for him. He hired an architect and used two different wrenches.";
                }
                if ((cumulative_visits % 200) == 168) {
                    cfg_custom_engine_name = "Fact: Astronauts are so much more efficient than you, that space dust is actually seen as a clear liquid by a telescope.";
                }
                if ((cumulative_visits % 200) == 169) {
                    cfg_custom_engine_name = "Fact: Manned space shuttles were originally manufactured for the military. They weren't sold to the public for over a decade.";
                }
                if ((cumulative_visits % 200) == 170) {
                    cfg_custom_engine_name = "Fact: While you wouldn't think a place so low on the world's food chain might be a good source of fish, it was. A catch of octopus.";
                }
                if ((cumulative_visits % 200) == 171) {
                    cfg_custom_engine_name = "Fact: The longest fully functional piece of machinery ever was only 98.6 miles. It was built for the 1962 Los Angeles Olympics and cost 3 million U.S. dollars. It was more than twice as long as the Empire State Building.";
                }
                if ((cumulative_visits % 200) == 172) {
                    cfg_custom_engine_name = "Fact: Most of NASA's lunar missions were conceived by Alan Shepard, the man who famously broke the sound barrier.";
                }
                if ((cumulative_visits % 200) == 173) {
                    cfg_custom_engine_name = "Fact: The First Full Moon of 2013 was named for Albert Einstein.";
                }
                if ((cumulative_visits % 200) == 174) {
                    cfg_custom_engine_name = "Fact: There are more than 1 million different kinds of ice, including almost all the known types of ice on Earth.";
                }
                if ((cumulative_visits % 200) == 175) {
                    cfg_custom_engine_name = "Fact: NASA tied their flags to a heavy stone on top of Mount Everest to prevent them from blowing away in the wind. It's a pretty impressive old-fashioned knot tie.";
                }
                if ((cumulative_visits % 200) == 176) {
                    cfg_custom_engine_name = "Fact: The first woman in space, Yuri Gagarin, spent 18 days in space in 1961.";
                }
                if ((cumulative_visits % 200) == 177) {
                    cfg_custom_engine_name = "Fact: An American pilot once claimed that people have crossed the Atlantic Ocean 2,450 times.";
                }
                if ((cumulative_visits % 200) == 178) {
                    cfg_custom_engine_name = "Fact: In 1945, the Sputnik spacecraft made the first documented satellite call.";
                }
                if ((cumulative_visits % 200) == 179) {
                    cfg_custom_engine_name = "Fact: The first transcontinental flight was completed in 1893.";
                }
                if ((cumulative_visits % 200) == 180) {
                    cfg_custom_engine_name = "Fact: If you put honey on the floor, a dog will walk on it.";
                }
                if ((cumulative_visits % 200) == 181) {
                    cfg_custom_engine_name = "Fact: Space helmets don't actually exist, except to create the impression that astronauts are in space.";
                }
                if ((cumulative_visits % 200) == 182) {
                    cfg_custom_engine_name = "Fact: The most expensive Olympic medals ever paid for were for 100 gold medals of a 2.4-pound rock on a nylon string. The total cost of all of the medals was 91 million dollars, or 183 million dollars including inflation.";
                }
                if ((cumulative_visits % 200) == 183) {
                    cfg_custom_engine_name = "Fact: The government claims to have discovered significant amounts of carbon, nitrogen and sulphur in the atmosphere. But they only know because NASA has made measurements over years of what's going up and down. So the level of all three gases is constantly changing.";
                }
                if ((cumulative_visits % 200) == 184) {
                    cfg_custom_engine_name = "Fact: Arizona currently produces more aluminum than its neighbors, China, United States, and Mexico combined.";
                }
                if ((cumulative_visits % 200) == 185) {
                    cfg_custom_engine_name = "Fact: The GPS satellite constellation that helped with our most recent GPS satellite fix in 2006 was launched from a cloud called Taurus 1, which orbits the sun once every 4.3 hours.";
                }
                if ((cumulative_visits % 200) == 186) {
                    cfg_custom_engine_name = "Fact: The first radio was invented in Germany during the Emancipation Proclamation.";
                }
                if ((cumulative_visits % 200) == 187) {
                    cfg_custom_engine_name = "Fact: It takes 59 million liters of water to produce one kilogram of gasoline.";
                }
                if ((cumulative_visits % 200) == 188) {
                    cfg_custom_engine_name = "Fact: In 1891, the word ''Easter'' first appeared in print in the Boston Globe. It was actually a misspelling of ''Ebenezer.''";
                }
                if ((cumulative_visits % 200) == 189) {
                    cfg_custom_engine_name = "Fact: Man's first message to the moon was actually a picture of a squirrel.";
                }
                if ((cumulative_visits % 200) == 190) {
                    cfg_custom_engine_name = "Fact: When a meteorite strikes Earth, it's not like being hit by lightning.";
                }
                if ((cumulative_visits % 200) == 191) {
                    cfg_custom_engine_name = "Fact: NASA has recorded 97 degrees Fahrenheit with temperatures as high as 98.4 F.";
                }
                if ((cumulative_visits % 200) == 192) {
                    cfg_custom_engine_name = "Fact: Just 0.1-inches of rain was recorded on our planet between 13,000 BC and 1600 AD. That's more rain than fell in the U.S. between 1850 and 2000.";
                }
                if ((cumulative_visits % 200) == 193) {
                    cfg_custom_engine_name = "Fact: The oldest known rocket on earth is a 70-year-old rocket in Tennessee. It was launched in 1899 by William H. Gass. He missed the moon and somehow missed England too. But he had more success launching his actual rocket than anyone ever.";
                }
                if ((cumulative_visits % 200) == 194) {
                    cfg_custom_engine_name = "Fact: More than 27 percent of all Americans can claim to have swum in a shark tank.";
                }
                if ((cumulative_visits % 200) == 195) {
                    cfg_custom_engine_name = "Fact: The entire human race lived at sea for the first eight centuries of their existence.";
                }
                if ((cumulative_visits % 200) == 196) {
                    cfg_custom_engine_name = "Fact: NASA and the U.S. Department of Energy have already approved more than 9 billion U.S. dollars in funding for innovative rockets that could ''melt'' space.";
                }
                if ((cumulative_visits % 200) == 197) {
                    cfg_custom_engine_name = "Fact: The fastest radio transmitters on earth are the Brazilian ones mounted on the back of a big white ibex.";
                }
                if ((cumulative_visits % 200) == 198) {
                    cfg_custom_engine_name = "Fact: Nothing is impossible. If France comes up with an e-reader, it'll happen. I mean, there's nothing stopping the French.";
                }
                if ((cumulative_visits % 200) == 199) {
                    cfg_custom_engine_name = "Fact: The most successful jacket has only three buttons.";
                }
                if ((cumulative_visits % 200) == 200) {
                    cfg_custom_engine_name = "Fact: Queen Elizabeth had been asked to serve in World War I, but declined. She said that serving in a war would be too distressing for a teenager. No mention is made of what military career she might have had pursued.";
                }

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
            if (cfg_nohandicap) {
                if (tmp == "place_free_handicap" || tmp == "set_free_handicap") {
                    gtp_printf(id, "false");
                    return 1;
                }
            }
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
            if (cfg_nohandicap) {
                if (s_commands[i] == "place_free_handicap" || s_commands[i] == "set_free_handicap") {
                    continue;
                }
            }
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
        movenum_now = 0; // Reset on new game
        win_message_sent = false; // Reset on new game
        win_message_confirmed_sent = false; // Reset on new game
        cfg_faster = false; // Reset on new game
        cfg_hyperspeed = false; // Reset on new game
        cfg_wearelosing = false; // Reset on new game
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

                cfg_handicapamount = game.get_handicap();
                if (game.get_handicap() >= 2) {
                    cfg_handicapgame = true;
                }

                if (game.get_handicap() <= 1) {
                    cfg_handicapgame = false;
                }

                if (game.get_handicap() >= 10) {
                    int move = FastBoard::RESIGN;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }

                if ((game.get_handicap() >= 2) && cfg_nohandicap) {
                    int move = FastBoard::RESIGN;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }

                if (game.get_komi() >= 9997.6f || game.get_komi() <= -9990.1f) {
                    int move = FastBoard::RESIGN;
                    game.play_move(move);
                    std::string vertex = game.move_to_text(move);
                    gtp_printf(id, "%s", vertex.c_str());
                    return true;
                }


                std::string rankquestionmark = "..\\" + cfg_kgsusername + "\\" "0.txt";
                std::string rank30k = "..\\" + cfg_kgsusername + "\\" "1.txt";
                std::string rank29k = "..\\" + cfg_kgsusername + "\\" "2.txt";
                std::string rank28k = "..\\" + cfg_kgsusername + "\\" "3.txt";
                std::string rank27k = "..\\" + cfg_kgsusername + "\\" "4.txt";
                std::string rank26k = "..\\" + cfg_kgsusername + "\\" "5.txt";
                std::string rank25k = "..\\" + cfg_kgsusername + "\\" "6.txt";
                std::string rank24k = "..\\" + cfg_kgsusername + "\\" "7.txt";
                std::string rank23k = "..\\" + cfg_kgsusername + "\\" "8.txt";
                std::string rank22k = "..\\" + cfg_kgsusername + "\\" "9.txt";
                std::string rank21k = "..\\" + cfg_kgsusername + "\\" "10.txt";
                std::string rank20k = "..\\" + cfg_kgsusername + "\\" "11.txt";
                std::string rank19k = "..\\" + cfg_kgsusername + "\\" "12.txt";
                std::string rank18k = "..\\" + cfg_kgsusername + "\\" "13.txt";
                std::string rank17k = "..\\" + cfg_kgsusername + "\\" "14.txt";
                std::string rank16k = "..\\" + cfg_kgsusername + "\\" "15.txt";
                std::string rank15k = "..\\" + cfg_kgsusername + "\\" "16.txt";
                std::string rank14k = "..\\" + cfg_kgsusername + "\\" "17.txt";
                std::string rank13k = "..\\" + cfg_kgsusername + "\\" "18.txt";
                std::string rank12k = "..\\" + cfg_kgsusername + "\\" "19.txt";
                std::string rank11k = "..\\" + cfg_kgsusername + "\\" "20.txt";
                std::string rank10k = "..\\" + cfg_kgsusername + "\\" "21.txt";
                std::string rank9k = "..\\" + cfg_kgsusername + "\\" "22.txt";
                std::string rank8k = "..\\" + cfg_kgsusername + "\\" "23.txt";
                std::string rank7k = "..\\" + cfg_kgsusername + "\\" "24.txt";
                std::string rank6k = "..\\" + cfg_kgsusername + "\\" "25.txt";
                std::string rank5k = "..\\" + cfg_kgsusername + "\\" "26.txt";
                std::string rank4k = "..\\" + cfg_kgsusername + "\\" "27.txt";
                std::string rank3k = "..\\" + cfg_kgsusername + "\\" "28.txt";
                std::string rank2k = "..\\" + cfg_kgsusername + "\\" "29.txt";
                std::string rank1k = "..\\" + cfg_kgsusername + "\\" "30.txt";
                std::string rank1d = "..\\" + cfg_kgsusername + "\\" "31.txt";
                std::string rank2d = "..\\" + cfg_kgsusername + "\\" "32.txt";
                std::string rank3d = "..\\" + cfg_kgsusername + "\\" "33.txt";
                std::string rank4d = "..\\" + cfg_kgsusername + "\\" "34.txt";
                std::string rank5d = "..\\" + cfg_kgsusername + "\\" "35.txt";
                std::string rank6d = "..\\" + cfg_kgsusername + "\\" "36.txt";
                std::string rank7d = "..\\" + cfg_kgsusername + "\\" "37.txt";
                std::string rank8d = "..\\" + cfg_kgsusername + "\\" "38.txt";
                std::string rank9d = "..\\" + cfg_kgsusername + "\\" "39.txt";
                if (boost::filesystem::exists(rankquestionmark)) {
                    //gtp_printf(id, "Unknown rank detected.");
                    cfg_opponentrank = 0;
                    boost::filesystem::remove(rankquestionmark);
                    resign_next = true;
                }
                if (boost::filesystem::exists(rank30k)) {
                    //gtp_printf(id, "30k detected.");
                    cfg_opponentrank = 1;
                    boost::filesystem::remove(rank30k);
                }
                if (boost::filesystem::exists(rank29k)) {
                    //gtp_printf(id, "29k detected.");
                    boost::filesystem::remove(rank29k);
                    cfg_opponentrank = 2;
                }
                if (boost::filesystem::exists(rank28k)) {
                    //gtp_printf(id, "28k detected.");
                    cfg_opponentrank = 3;
                    boost::filesystem::remove(rank28k);
                }
                if (boost::filesystem::exists(rank27k)) {
                    //gtp_printf(id, "27k detected.");
                    cfg_opponentrank = 4;
                    boost::filesystem::remove(rank27k);
                }
                if (boost::filesystem::exists(rank26k)) {
                    //gtp_printf(id, "26k detected.");
                    cfg_opponentrank = 5;
                    boost::filesystem::remove(rank26k);
                }
                if (boost::filesystem::exists(rank25k)) {
                    //gtp_printf(id, "25k detected.");
                    cfg_opponentrank = 6;
                    boost::filesystem::remove(rank25k);
                }
                if (boost::filesystem::exists(rank24k)) {
                    //gtp_printf(id, "24k detected.");
                    cfg_opponentrank = 7;
                    boost::filesystem::remove(rank24k);
                }
                if (boost::filesystem::exists(rank23k)) {
                    //gtp_printf(id, "23k detected.");
                    cfg_opponentrank = 8;
                    boost::filesystem::remove(rank23k);
                }
                if (boost::filesystem::exists(rank22k)) {
                    //gtp_printf(id, "22k detected.");
                    cfg_opponentrank = 9;
                    boost::filesystem::remove(rank22k);
                }
                if (boost::filesystem::exists(rank21k)) {
                    //gtp_printf(id, "21k detected.");
                    cfg_opponentrank = 10;
                    boost::filesystem::remove(rank21k);
                }
                if (boost::filesystem::exists(rank20k)) {
                    //gtp_printf(id, "20k detected.");
                    cfg_opponentrank = 11;
                    boost::filesystem::remove(rank20k);
                }
                if (boost::filesystem::exists(rank19k)) {
                    //gtp_printf(id, "19k detected.");
                    cfg_opponentrank = 12;
                    boost::filesystem::remove(rank19k);
                }
                if (boost::filesystem::exists(rank18k)) {
                    //gtp_printf(id, "18k detected.");
                    cfg_opponentrank = 13;
                    boost::filesystem::remove(rank18k);
                }
                if (boost::filesystem::exists(rank17k)) {
                    //gtp_printf(id, "17k detected.");
                    cfg_opponentrank = 14;
                    boost::filesystem::remove(rank17k);
                }
                if (boost::filesystem::exists(rank16k)) {
                    //gtp_printf(id, "16k detected.");
                    cfg_opponentrank = 15;
                    boost::filesystem::remove(rank16k);
                }
                if (boost::filesystem::exists(rank15k)) {
                    //gtp_printf(id, "15k detected.");
                    cfg_opponentrank = 16;
                    boost::filesystem::remove(rank15k);
                }
                if (boost::filesystem::exists(rank14k)) {
                    //gtp_printf(id, "14k detected.");
                    cfg_opponentrank = 17;
                    boost::filesystem::remove(rank14k);
                }
                if (boost::filesystem::exists(rank13k)) {
                    //gtp_printf(id, "13k detected.");
                    cfg_opponentrank = 18;
                    boost::filesystem::remove(rank13k);
                }
                if (boost::filesystem::exists(rank12k)) {
                    //gtp_printf(id, "12k detected.");
                    cfg_opponentrank = 19;
                    boost::filesystem::remove(rank12k);
                }
                if (boost::filesystem::exists(rank11k)) {
                    //gtp_printf(id, "11k detected.");
                    cfg_opponentrank = 20;
                    boost::filesystem::remove(rank11k);
                }
                if (boost::filesystem::exists(rank10k)) {
                    //gtp_printf(id, "10k detected.");
                    cfg_opponentrank = 21;
                    boost::filesystem::remove(rank10k);
                }
                if (boost::filesystem::exists(rank9k)) {
                    //gtp_printf(id, "9k detected.");
                    cfg_opponentrank = 22;
                    boost::filesystem::remove(rank9k);
                }
                if (boost::filesystem::exists(rank8k)) {
                    //gtp_printf(id, "8k detected.");
                    cfg_opponentrank = 23;
                    boost::filesystem::remove(rank8k);
                }
                if (boost::filesystem::exists(rank7k)) {
                    //gtp_printf(id, "7k detected.");
                    cfg_opponentrank = 24;
                    boost::filesystem::remove(rank7k);
                }
                if (boost::filesystem::exists(rank6k)) {
                    //gtp_printf(id, "6k detected.");
                    cfg_opponentrank = 25;
                    boost::filesystem::remove(rank6k);
                }
                if (boost::filesystem::exists(rank5k)) {
                    //gtp_printf(id, "5k detected.");
                    cfg_opponentrank = 26;
                    boost::filesystem::remove(rank5k);
                }
                if (boost::filesystem::exists(rank4k)) {
                    //gtp_printf(id, "4k detected.");
                    cfg_opponentrank = 27;
                    boost::filesystem::remove(rank4k);
                }
                if (boost::filesystem::exists(rank3k)) {
                    //gtp_printf(id, "3k detected.");
                    cfg_opponentrank = 28;
                    boost::filesystem::remove(rank3k);
                }
                if (boost::filesystem::exists(rank2k)) {
                    //gtp_printf(id, "2k detected.");
                    cfg_opponentrank = 29;
                    boost::filesystem::remove(rank2k);
                }
                if (boost::filesystem::exists(rank1k)) {
                    //gtp_printf(id, "1k detected.");
                    cfg_opponentrank = 30;
                    boost::filesystem::remove(rank1k);
                }
                if (boost::filesystem::exists(rank1d)) {
                    //gtp_printf(id, "1d detected.");
                    cfg_opponentrank = 31;
                    boost::filesystem::remove(rank1d);
                }
                if (boost::filesystem::exists(rank2d)) {
                    //gtp_printf(id, "2d detected.");
                    cfg_opponentrank = 32;
                    boost::filesystem::remove(rank2d);
                }
                if (boost::filesystem::exists(rank3d)) {
                    //gtp_printf(id, "3d detected.");
                    cfg_opponentrank = 33;
                    boost::filesystem::remove(rank3d);
                }
                if (boost::filesystem::exists(rank4d)) {
                    //gtp_printf(id, "4d detected.");
                    cfg_opponentrank = 34;
                    boost::filesystem::remove(rank4d);
                }
                if (boost::filesystem::exists(rank5d)) {
                    //gtp_printf(id, "5d detected.");
                    cfg_opponentrank = 35;
                    boost::filesystem::remove(rank5d);
                }
                if (boost::filesystem::exists(rank6d)) {
                    //gtp_printf(id, "6d detected.");
                    cfg_opponentrank = 36;
                    boost::filesystem::remove(rank6d);
                }
                if (boost::filesystem::exists(rank7d)) {
                    //gtp_printf(id, "7d detected.");
                    cfg_opponentrank = 37;
                    boost::filesystem::remove(rank7d);
                }
                if (boost::filesystem::exists(rank8d)) {
                    //gtp_printf(id, "8d detected.");
                    cfg_opponentrank = 38;
                    boost::filesystem::remove(rank8d);
                }
                if (boost::filesystem::exists(rank9d)) {
                    //gtp_printf(id, "9d detected.");
                    cfg_opponentrank = 39;
                    boost::filesystem::remove(rank9d);
                }

                if (cfg_opponentrank > cfg_maxrankallowed) {
                    resign_next = true;
                }

                if (cfg_opponentrank < cfg_minrankallowed) {
                    resign_next = true;
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
        if (px == "x0") {
            cfg_custom_engine_name = "";
            cmdstream >> word;
            do {
                cfg_custom_engine_name += word;
                cfg_custom_engine_name += " ";
                cmdstream >> word;
            } while (!cmdstream.fail());
        }
        if (px == "x1") {
            cmdstream >> word;
            if (word == "pass") {
                pass_next = true;
            }

            if (word == "crossbot_enable") {
                cfg_crossbot = true;
            }
            if (word == "crossbot_disable") {
                cfg_crossbot = false;
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

            if (word == "weirdbot_enable") {
                cfg_weirdbot = true;
            }
            if (word == "weirdbot_disable") {
                cfg_weirdbot = false;
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
            if (word == "hyperspeed_enable") {
                cfg_hyperspeed = true;
            }
            if (word == "hyperspeed_disable") {
                cfg_hyperspeed = false;
            }
            if (word == "tenukibot_enable") {
                cfg_tenukibot = true;
            }
            if (word == "tenukibot_disable") {
                cfg_tenukibot = false;
            }
            if (word == "followbot_enable") {
                cfg_followbot = true;
            }
            if (word == "followbot_disable") {
                cfg_followbot = false;
            }
            if (word == "superslow_enable") {
                cfg_superslow = true;
            }
            if (word == "superslow_disable") {
                cfg_superslow = false;
            }
            if (word == "rmtb_enable") {
                cfg_rankmatchingtiebot = true;
            }
            if (word == "rmtb_disable") {
                cfg_rankmatchingtiebot = false;
            }
            if (word == "ponder") {
                cfg_allow_pondering = true;
            }
            if (word == "noponder") {
                cfg_allow_pondering = false;
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
        movenum_now = 0; // Reset on new game
        win_message_sent = false; // Reset on new game
        win_message_confirmed_sent = false; // Reset on new game
        cfg_faster = false; // Reset on new game
        cfg_hyperspeed = false; // Reset on new game
        cfg_wearelosing = false; // Reset on new game
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
