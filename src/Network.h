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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "config.h"

#include <array>
#include <bitset>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "FastState.h"
#include "GameState.h"
#include "OpenCLScheduler.h"

class NNCache;

class Network {
public:
    enum Ensemble {
        DIRECT, RANDOM_ROTATION
    };
    using BoardPlane = std::bitset<19*19>;
    using NNPlanes = std::vector<BoardPlane>;
    using scored_node = std::pair<float, int>;
    using Netresult = std::pair<std::vector<scored_node>, float>;

    Netresult get_scored_moves(const GameState* state,
                                      Ensemble ensemble,
                                      int rotation = -1,
                                      bool skip_cache = false);
    // File format version
    static constexpr auto FORMAT_VERSION = 1;
    static constexpr auto INPUT_MOVES = 8;
    static constexpr auto INPUT_CHANNELS = 2 * INPUT_MOVES + 2;
    static constexpr auto OUTPUTS_POLICY = 2;
    static constexpr auto OUTPUTS_VALUE = 1;

    // Winograd filter transformation changes 3x3 filters to 4x4
    static constexpr auto WINOGRAD_ALPHA = 4;
    static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;

    void initialize(char* weights_file, char* tuning_file);
    void benchmark(const GameState * state, int iterations = 1600);
    void show_heatmap(const FastState * state, Netresult & netres,
                             bool topmoves);
    void softmax(const std::vector<float>& input,
                        std::vector<float>& output,
                        float temperature = 1.0f);

    void gather_features(const GameState* state, NNPlanes& planes);
private:
    std::pair<int, int> load_v1_network(std::ifstream& wtfile);
    std::pair<int, int> load_network_file(std::string filename);
    void process_bn_var(std::vector<float>& weights,
                               const float epsilon=1e-5f);

    std::vector<float> winograd_transform_f(const std::vector<float>& f,
        const int outputs, const int channels);
    std::vector<float> zeropad_U(const std::vector<float>& U,
        const int outputs, const int channels,
        const int outputs_pad, const int channels_pad);
    void winograd_transform_in(const std::vector<float>& in,
                                      std::vector<float>& V,
                                      const int C);
    void winograd_transform_out(const std::vector<float>& M,
                                       std::vector<float>& Y,
                                       const int K);
    void winograd_convolve3(const int outputs,
                                   const std::vector<float>& input,
                                   const std::vector<float>& U,
                                   std::vector<float>& V,
                                   std::vector<float>& M,
                                   std::vector<float>& output);
    void winograd_sgemm(const std::vector<float>& U,
                               std::vector<float>& V,
                               std::vector<float>& M, const int C, const int K);
    int rotate_nn_idx(const int vertex, int symmetry);
    void fill_input_plane_pair(
      const FullBoard& board, BoardPlane& black, BoardPlane& white);
    Netresult get_scored_moves_internal(
      const GameState* state, NNPlanes & planes, int rotation);
#if defined(USE_BLAS)
    void forward_cpu(std::vector<float>& input,
                            std::vector<float>& output_pol,
                            std::vector<float>& output_val);

#endif

	// Input + residual block tower
	std::vector<std::vector<float>> conv_weights;
	std::vector<std::vector<float>> conv_biases;
	std::vector<std::vector<float>> batchnorm_means;
	std::vector<std::vector<float>> batchnorm_stddivs;

	// Policy head
	std::vector<float> conv_pol_w;
	std::vector<float> conv_pol_b;
	std::array<float, 2> bn_pol_w1;
	std::array<float, 2> bn_pol_w2;

	std::array<float, 261364> ip_pol_w;
	std::array<float, 362> ip_pol_b;

	// Value head
	std::vector<float> conv_val_w;
	std::vector<float> conv_val_b;
	std::array<float, 1> bn_val_w1;
	std::array<float, 1> bn_val_w2;

	std::array<float, 92416> ip1_val_w;
	std::array<float, 256> ip1_val_b;

	std::array<float, 256> ip2_val_w;
	std::array<float, 1> ip2_val_b;

	// Rotation helper
	std::array<std::array<int, 361>, 8> rotate_nn_idx_table;
	OpenCLScheduler opencl;
	NNCache *nncache;// &get_NNCache(void);
};

extern Network net_6b;
extern Network net_20b;

#endif
