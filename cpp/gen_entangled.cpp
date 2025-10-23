#include <cmath>
#include <vector>
#include <string>
#include "stim/circuit/circuit.h"
#include "stim/simulators/tableau_simulator.h"
#include <iostream>
#include <cstdio>
using namespace std;

struct Depols { double p1, p2, pprep, pmeas, p_idle_equiv; };

Depols depol_from_delay(double delay_s, double T1=200e-6, double T2=150e-6, double p_base=1e-3) {
    double p_idle = (3 - exp(-delay_s/T1) - 2*exp(-delay_s/T2)) / 2.0;
    double p_eff = 1 - (1 - p_base) * (1 - p_idle);
    return {p_eff, p_eff, p_eff, p_eff, p_idle};
}

double calc_epr_delay(double L_m, int num_pairs, int cap, double ent_rate_hz) {
    if (num_pairs <= 0) return 0.0;
    double p_succ = pow(10.0, -0.02 * L_m / 1000.0);
    if (p_succ <= 0 || cap <= 0 || ent_rate_hz <= 0) return INFINITY;
    int rounds = (num_pairs + cap - 1) / cap;
    int per_round = min(num_pairs, cap);
    return rounds * (per_round / (p_succ * ent_rate_hz));
}
double calc_round_delay(double L_m, int num_pairs, int cap, double ent_rate_hz, double meas_delay_s) {
    return max(meas_delay_s, calc_epr_delay(L_m, num_pairs, cap, ent_rate_hz));
}

static inline vector<pair<uint32_t,uint32_t>> pairwise(const vector<uint32_t>& qs) {
    if (qs.size() % 2) throw runtime_error("odd number of targets");
    vector<pair<uint32_t,uint32_t>> out;
    for (size_t i=0;i<qs.size();i+=2) out.emplace_back(qs[i], qs[i+1]);
    return out;
}

static inline std::vector<uint32_t> qubit_targets(stim::SpanRef<const stim::GateTarget> ts) {
    std::vector<uint32_t> qs;
    for (auto *ptr = ts.ptr_start; ptr < ts.ptr_end; ptr++) {
        if (ptr->is_qubit_target()) {
            qs.push_back(ptr->qubit_value());
        }
    }
    return qs;
}

stim::Circuit gen_entangled_circuit(const stim::Circuit& src, int d, int r, int center_line=-1) {
    // parameters
    double meas_delay_s = 1e-9;  // adjust to your units
    double ent_rate_hz  = 1e6;
    int    channel_cap  = 1000;
    double channel_L_m  = 5;
    double mult_ent_cnot = 5.0;

    double delay = calc_round_delay(channel_L_m, d, channel_cap, ent_rate_hz, meas_delay_s);
    auto dep = depol_from_delay(delay);
    double p1 = dep.p1, p2 = dep.p2, pprep = dep.pprep, pmeas = dep.pmeas;

    stim::Circuit dst;

    // (A) find seam qubits from coordinates, if available
    // In C++ you can parse coordinates by reading the circuit text or maintaining your own map.
    // If you have coords in src, consider extracting them in Python and passing the set into C++.
    // Here we assume `center_qubits` provided/known:
    auto is_on_seam = [&](uint32_t q)->bool { /* TODO: implement or pass in */ return false; };

    // (B) recursive copier with injections
    std::function<void(const stim::Circuit&, stim::Circuit&)> process;
    process = [&](const stim::Circuit &s, stim::Circuit &out) {
        for (const auto &inst : src.operations) {
            if (inst.gate_type == stim::GateType::REPEAT) {
                uint64_t reps = inst.repeat_block_rep_count();
                stim::Circuit body_out;
                process(inst.repeat_block_body(src), body_out);
                dst.append_repeat_block(reps, body_out, "");  // empty tag ok
                continue;
            }

            std::vector<uint32_t> qs = qubit_targets(inst.targets);

            // Pre-measurement depol
            if ((inst.gate_type == stim::GateType::M ||
                 inst.gate_type == stim::GateType::MR) &&
                pmeas > 0 && !qs.empty()) {
                out.safe_append_ua("DEPOLARIZE1", qs, pmeas);
            }

            // Original op
            out.safe_append(inst, /*block_fusion=*/false);

            // Reset depol
            if ((inst.gate_type == stim::GateType::R ||
                 inst.gate_type == stim::GateType::RX ||
                 inst.gate_type == stim::GateType::RY) &&
                pprep > 0 && !qs.empty()) {
                out.safe_append_ua("DEPOLARIZE1", qs, pprep);
            }

            // One-qubit gates
            if ((inst.gate_type == stim::GateType::H ||
                 inst.gate_type == stim::GateType::S ||
                 inst.gate_type == stim::GateType::S_DAG ||
                 inst.gate_type == stim::GateType::X ||
                 inst.gate_type == stim::GateType::Y ||
                 inst.gate_type == stim::GateType::Z ||
                 inst.gate_type == stim::GateType::SQRT_X ||
                 inst.gate_type == stim::GateType::SQRT_X_DAG ||
                 inst.gate_type == stim::GateType::SQRT_Y ||
                 inst.gate_type == stim::GateType::SQRT_Y_DAG) &&
                p1 > 0 && !qs.empty()) {
                out.safe_append_ua("DEPOLARIZE1", qs, p1);
            }

            // Two-qubit gates (pairwise)
            if ((inst.gate_type == stim::GateType::CZ ||
                 inst.gate_type == stim::GateType::SWAP ||
                 inst.gate_type == stim::GateType::ISWAP) &&
                p2 > 0 && qs.size() >= 2) {
                for (size_t i = 0; i + 1 < qs.size(); i += 2) {
                    out.safe_append_ua("DEPOLARIZE2", {qs[i], qs[i + 1]}, p2);
                }
            }
        }
    };

    process(src, dst);
    return dst;
}

// int main() {
//     stim::Circuit c("H 0\nCNOT 0 1\nM 0 1\n");
//     std::mt19937_64 rng(0);
//     stim::TableauSimulator<64> sim(std::move(rng), /*num_qubits=*/2);
//     sim.safe_do_circuit(c, 1);
//     std::cout << "Ran successfully!\n";
// }

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