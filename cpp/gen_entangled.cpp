#include <cmath>
#include <vector>
#include <string>
#include "stim/circuit/circuit.h"
#include "stim/simulators/tableau_simulator.h"
#include <iostream>
#include <cstdio>
using namespace std;

// from the tannu paper
std::vector<double> depol_from_delay(double tau_s, double T1=200e-6, double T2=150e-6, double p_base=1e-3) {
    double p_x = (1 - exp(-tau_s/T1))/4.0;
    double p_y = p_x;
    double p_z = (1 - exp(-tau_s/T2))/2.0 - p_x;

    return {p_x, p_y, p_z};
}

// delay (in seconds) due to EPR pair generation rate
double calc_epr_delay_s(double channel_length_m, int req_num_epr_pairs, int channel_capacity, double entanglement_rate_hz) {
    // edge case when it's only superstabilizers
    if (req_num_epr_pairs <= 0) return 0.0;

    double attenuation_success_prob = pow(10.0, -0.02 * channel_length_m / 1000.0);

    // other edge cases
    if (attenuation_success_prob <= 0 || channel_capacity <= 0 || entanglement_rate_hz <= 0) return INFINITY;

    // number of epr pairs that will go thru the channel
    int epr_pairs_per_round = min(req_num_epr_pairs, channel_capacity);

    // # rounds to get all the required EPR pairs thru the channel
    // ASSUMPTION: this is always 1 since the channel capacity is so big
    int num_epr_pair_distribution_rounds = std::ceil(req_num_epr_pairs / channel_capacity);
    double delay_per_round_s = epr_pairs_per_round / (attenuation_success_prob * entanglement_rate_hz);

    // return total delay
    return delay_per_round_s * num_epr_pair_distribution_rounds;
}

std::vector<uint32_t> get_all_qubits(const stim::Circuit &circuit) {
    std::set<uint32_t> qubit_set;

    for (const auto &op : circuit.operations) {
        for (const auto &target : op.targets) {
            if (target.is_qubit_target()) {
                qubit_set.insert(target.qubit_value());
            }
        }
    }

    // Convert the set to a vector
    return std::vector<uint32_t>(qubit_set.begin(), qubit_set.end());
}


stim::Circuit gen_entangled_circuit(const stim::Circuit& src, int d, int r, int num_superstabilizers){//, int center_line=-1) {
    // parameters
    double meas_delay_s = 1e-9;  // adjust to your units
    double ent_rate_hz  = 1e6;
    int    channel_cap  = 1000;
    double channel_length_m  = 5;
    double entanglement_cnot_multiplier = 5.0;

    // calc delay
    int req_num_epr_pairs = d - num_superstabilizers;
    double epr_delay_s = calc_epr_delay_s(channel_length_m, req_num_epr_pairs, channel_cap, ent_rate_hz);
    double delay = max(epr_delay_s, meas_delay_s);

    // calc depol rates
    auto dep = depol_from_delay(delay);

    stim::Circuit dst;
    auto all_qubits = get_all_qubits(src);

    // recursive thing to get all the repeat blocks
    std::function<void(const stim::Circuit&, stim::Circuit&)> process;
    process = [&](const stim::Circuit &s, stim::Circuit &out) {
        for (const auto &inst : src.operations) {
            if (inst.gate_type == stim::GateType::REPEAT) {
                uint64_t reps = inst.repeat_block_rep_count();
                stim::Circuit body_out;

                // important: add delay depol to all qubits at the start of each round.
                // doesn't really matter for check qubits since they get reset
                // but for data it could introduce logical errors.
                body_out.safe_append_u("PAULI_CHANNEL_1", all_qubits, dep);
                process(inst.repeat_block_body(src), body_out);
                dst.append_repeat_block(reps, body_out, "");  // empty tag ok
                continue;
            }

            // Original op
            out.safe_append(inst, false);
        }
    };

    process(src, dst);
    return dst;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_circuit.stim>\n";
        return 1;
    }

    const char *path = argv[1];
    FILE *f = fopen(path, "r");
    if (!f) {
        std::cerr << "Error: could not open file '" << path << "'\n";
        return 1;
    }

    try {
        stim::Circuit circuit = stim::Circuit::from_file(f);
        fclose(f);  // important: close after loading

        std::cout << "Loaded circuit with "
                  << circuit.count_qubits() << " qubits\n.";

        std::mt19937_64 rng(0);
        stim::TableauSimulator<64> sim(std::move(rng));
        sim.safe_do_circuit(circuit, 1);

        std::cout << "Simulation complete!\n";
    } catch (const std::exception &ex) {
        std::cerr << "Stim error: " << ex.what() << "\n";
        fclose(f);
        return 1;
    }

    return 0;
}