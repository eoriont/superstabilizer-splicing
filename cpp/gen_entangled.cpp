#include <cmath>
#include <vector>
#include <string>
#include "stim/circuit/circuit.h"
#include "stim/simulators/tableau_simulator.h"
#include <iostream>
#include <cstdio>
#include <fstream>
#include <optional>
#include <unordered_set>
#include <array>
#include <set>
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

struct CircuitSettings {
    double measurement_delay_s = 1e-9;
    double entanglement_rate_hz = 1e6;
    int    channel_capacity = 1000;
    double channel_length_m = 5;
    double entanglement_cnot_multiplier = 5.0;
    double depol1 = 0.005;
    double depol2 = 0.01;
};

/**
 * Qubit helpers
 */
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

std::unordered_set<uint32_t> make_seam_qubits(const stim::Circuit &circuit, int d) {
    std::unordered_set<uint32_t> seam_qubits;

    // Get final coordinates for all qubits
    auto coords = circuit.get_final_qubit_coords();

    for (uint32_t q = 0; q < coords.size(); q++) {
        const auto &coord = coords[q];
        if (coord.size() != 2) continue;  // skip malformed coords

        double x = coord[1];  // X coordinate
        // Check if x equals d (within a small tolerance, in case of floats)
        if (std::abs(x - d) < 1e-9) {
            seam_qubits.insert(q);
        }
    }

    return seam_qubits;
}

std::vector<uint32_t> find_deform_qubits(const stim::Circuit &circuit, int d) {
    auto coords = circuit.get_final_qubit_coords();

    const size_t limit = std::min<size_t>(coords.size(), size_t(d) * size_t(d));
    std::vector<uint32_t> deform;
    deform.reserve(limit);

    for (uint32_t q = 0; q < limit; ++q) {
        if (coords[q].empty()) {
            deform.push_back(q);
        }
    }
    return deform;
}


int infer_surface_code_distance(const stim::Circuit &circuit) {
    auto coords = circuit.get_final_qubit_coords();
    std::set<int> xs, ys;

    for (const auto &c : coords) {
        int x = static_cast<int>(std::round(c.second[0]));
        int y = static_cast<int>(std::round(c.second[1]));
        // data qubits are always at even coords
        if (x % 2 == 0 && y % 2 == 0) {
            xs.insert(x);
            ys.insert(y);
        }
    }

    if (xs.empty() || ys.empty()) {
        throw std::runtime_error("No qubit coordinates found in circuit.");
    }

    int d_x = static_cast<int>(xs.size());
    int d_y = static_cast<int>(ys.size());
    return std::min(d_x, d_y);
}

/**
 * Circuit function!
 */
stim::Circuit gen_entangled_circuit(const stim::Circuit& src, CircuitSettings cfg) {
    // this should also be the # of superstabilizers, assuming
    // that no deform is touching.
    int d = infer_surface_code_distance(src);
    auto deform_qubits = find_deform_qubits(src, d);
    int num_superstabilizers = deform_qubits.size();
    cout << "(d=" << d << ") and num_superstabilizers = " << num_superstabilizers << "\n";

    // calc delay
    int req_num_epr_pairs = d - num_superstabilizers;
    double epr_delay_s = calc_epr_delay_s(cfg.channel_length_m, req_num_epr_pairs, cfg.channel_capacity, cfg.entanglement_rate_hz);
    double delay = max(epr_delay_s, cfg.measurement_delay_s);

    // calc depol rates (px,py,pz)
    auto dep = depol_from_delay(delay);

    stim::Circuit dst;
    auto all_qubits = get_all_qubits(src);
    bool uses_repeat_blocks = false;

    std::unordered_set<uint32_t> seam_qubits = make_seam_qubits(src, d);

    // recursive thing to get all the repeat blocks
    std::function<void(const stim::Circuit&, stim::Circuit&)> process;
    process = [&](const stim::Circuit &src, stim::Circuit &dst) {
        stim::GateType last1 = stim::GateType::NOT_A_GATE;
        stim::GateType last2 = stim::GateType::NOT_A_GATE;
        for (const auto &inst : src.operations) {
            if (inst.gate_type == stim::GateType::REPEAT) {
                // TODO: get # rounds here. Should be the parameter + 1 for the starting round
                uses_repeat_blocks = true;
                uint64_t reps = inst.repeat_block_rep_count();
                stim::Circuit body_out;

                // important: add delay depol to all qubits at the start of each round.
                // doesn't really matter for check qubits since they get reset
                // but for data it could introduce logical errors.
                body_out.safe_append_u("PAULI_CHANNEL_1", all_qubits, dep);

                // recurse
                process(inst.repeat_block_body(src), body_out);
                dst.append_repeat_block(reps, body_out, "");  // empty tag ok
                continue;
            }

            // this is for the second subround for the superstabilizers.
            if (num_superstabilizers > 0) {
                if (inst.gate_type == stim::GateType::R &&
                    last1 == stim::GateType::TICK &&
                    last2 == stim::GateType::DETECTOR) {
                    dst.safe_append_u("PAULI_CHANNEL_1", all_qubits, dep);
                }
            }

            // For the CNOT depol2's, we use the multiplier if it is a remote CNOT
            if (inst.gate_type == stim::GateType::DEPOLARIZE2) {
                std::vector<uint32_t> qs = qubit_targets(inst.targets);
                auto pairs = pairwise(qs);
                std::vector<uint32_t> affected;
                std::vector<uint32_t> unaffected;

                for (auto &[q0, q1] : pairs) {
                    bool hits_seam = seam_qubits.count(q0) || seam_qubits.count(q1);
                    if (hits_seam) {
                        affected.push_back(q0);
                        affected.push_back(q1);
                    } else {
                        unaffected.push_back(q0);
                        unaffected.push_back(q1);
                    }
                }

                // Emit unaffected pairs with original p
                if (!unaffected.empty()) {
                    dst.safe_append_ua("DEPOLARIZE2", unaffected, cfg.depol2);
                }
                // Emit affected pairs with scaled p
                if (!affected.empty()) {
                    dst.safe_append_ua("DEPOLARIZE2", affected, cfg.depol2 * cfg.entanglement_cnot_multiplier);
                }
                continue;
            }

            // For depol1's, we just replace the prob
            if (inst.gate_type == stim::GateType::DEPOLARIZE1) {
                std::vector<uint32_t> qs = qubit_targets(inst.targets);

                // Emit one new DEPOLARIZE1 instruction with the new probability
                dst.safe_append_ua("DEPOLARIZE1", qs, cfg.depol1);
                continue;
            }

            // Original op
            dst.safe_append(inst, false);

            last2 = last1;
            last1 = inst.gate_type;
        }
    };

    process(src, dst);

    if (!uses_repeat_blocks) {
        throw std::runtime_error("No repeat blocks found. Input stim file must use repeat blocks for the error injection to work.");
    }
    return dst;
}

/**
 * Command line interface stuff
 */

static std::string default_out_path_for(const std::string &in) {
    // Insert _ent before ".stim" if present, else append "_ent.stim"
    auto pos = in.rfind(".stim");
    if (pos != std::string::npos && pos == in.size() - 5) {
        return in.substr(0, pos) + "_ent.stim";
    }
    return in + "_ent.stim";
}

static bool parse_double(const char *s, double &val) {
    char *end = nullptr;
    double v = std::strtod(s, &end);
    if (!end || *end != '\0') return false;
    val = v;
    return true;
}

static bool parse_int(const char *s, int &val) {
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    val = static_cast<int>(v);
    return true;
}

static void print_usage(const char *prog) {
    std::cerr <<
        "Usage: " << prog << " <path_to_circuit.stim> [options]\n"
        "Options:\n"
        "  --measurement-delay-s <double>         (default 1e-9)\n"
        "  --entanglement-rate-hz <double>        (default 1e6)\n"
        "  --depol1 <double>                      (default 0.005)\n"
        "  --depol2 <double>                      (default 0.01)\n"
        "  --channel-capacity <int>               (default 1000)\n"
        "  --channel-length-m <double>            (default 5.0)\n"
        "  --entanglement-cnot-multiplier <double>(default 5.0)\n"
        "  -o, --output <path>                    (default: input with _ent.stim)\n"
        "  -h, --help\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // required input path may be first non-flag argument
    std::string input_path;
    CircuitSettings cfg;
    std::string out_path;

    // simple flag parsing
    for (int i = 1; i < argc; ) {
        std::string arg = argv[i];

        auto need_val = [&](const char *name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--measurement-delay-s") {
            const char *v = need_val(arg.c_str());
            if (!parse_double(v, cfg.measurement_delay_s)) { std::cerr << "Bad double: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "--entanglement-rate-hz") {
            const char *v = need_val(arg.c_str());
            if (!parse_double(v, cfg.entanglement_rate_hz)) { std::cerr << "Bad double: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "--depol1") {
            const char *v = need_val(arg.c_str());
            if (!parse_double(v, cfg.depol1)) { std::cerr << "Bad double: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "--depol2") {
            const char *v = need_val(arg.c_str());
            if (!parse_double(v, cfg.depol2)) { std::cerr << "Bad double: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "--channel-capacity") {
            const char *v = need_val(arg.c_str());
            if (!parse_int(v, cfg.channel_capacity)) { std::cerr << "Bad int: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "--channel-length-m") {
            const char *v = need_val(arg.c_str());
            if (!parse_double(v, cfg.channel_length_m)) { std::cerr << "Bad double: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "--entanglement-cnot-multiplier") {
            const char *v = need_val(arg.c_str());
            if (!parse_double(v, cfg.entanglement_cnot_multiplier)) { std::cerr << "Bad double: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "-o" || arg == "--output") {
            out_path = need_val(arg.c_str());
            ++i;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            return 2;
        } else {
            // first non-flag becomes input path
            if (input_path.empty()) {
                input_path = arg;
                ++i;
            } else {
                std::cerr << "Unexpected extra argument: " << arg << "\n";
                return 2;
            }
        }
    }

    if (input_path.empty()) {
        std::cerr << "Missing <path_to_circuit.stim>\n";
        print_usage(argv[0]);
        return 1;
    }
    if (out_path.empty()) {
        out_path = default_out_path_for(input_path);
    }

    FILE *f = std::fopen(input_path.c_str(), "r");
    if (!f) {
        std::cerr << "Error: could not open file '" << input_path << "'\n";
        return 1;
    }

    try {
        stim::Circuit circuit = stim::Circuit::from_file(f);
        std::fclose(f);  // close after loading

        std::cout << "Loaded circuit with "
                  << circuit.count_qubits() << " qubits.\n";

        stim::Circuit new_circ = gen_entangled_circuit(circuit, cfg);

        // Save to file
        std::ofstream out(out_path);
        if (!out) {
            std::cerr << "Error: could not open output file '" << out_path << "' for writing\n";
            return 1;
        }
        out << new_circ;
        out.close();

        std::cout << "Wrote entangled circuit to: " << out_path << "\n";
    } catch (const std::exception &ex) {
        std::cerr << "Stim error: " << ex.what() << "\n";
        std::fclose(f);  // in case exception before fclose above
        return 1;
    }

    return 0;
}
