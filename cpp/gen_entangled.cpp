#include <cmath>
#include <vector>
#include <string>
#include "stim/circuit/circuit.h"
#include "stim/simulators/tableau_simulator.h"
#include <iostream>
#include <cstdio>
#include <fstream>
#include <optional>
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
    int d = 5;
    int r = 15;
    int num_superstabilizers = 0;
};

stim::Circuit gen_entangled_circuit(const stim::Circuit& src, CircuitSettings cfg) {
    // calc delay
    int req_num_epr_pairs = cfg.d - cfg.num_superstabilizers;
    double epr_delay_s = calc_epr_delay_s(cfg.channel_length_m, req_num_epr_pairs, cfg.channel_capacity, cfg.entanglement_rate_hz);
    double delay = max(epr_delay_s, cfg.measurement_delay_s);

    // calc depol rates (px,py,pz)
    auto dep = depol_from_delay(delay);

    stim::Circuit dst;
    auto all_qubits = get_all_qubits(src);
    bool uses_repeat_blocks = false;

    // recursive thing to get all the repeat blocks
    std::function<void(const stim::Circuit&, stim::Circuit&)> process;
    process = [&](const stim::Circuit &src, stim::Circuit &dst) {
        for (const auto &inst : src.operations) {
            if (inst.gate_type == stim::GateType::REPEAT) {
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

            // Original op
            dst.safe_append(inst, false);
        }
    };

    process(src, dst);

    if (!uses_repeat_blocks) {
        throw std::runtime_error("No repeat blocks found. Input stim file must use repeat blocks for the error injection to work.");
    }
    return dst;
}

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
        "  --channel-capacity <int>               (default 1000)\n"
        "  --channel-length-m <double>            (default 5.0)\n"
        "  --entanglement-cnot-multiplier <double>(default 5.0)\n"
        "  --d <int>                              (default 5)\n"
        "  --r <int>                              (default 15)\n"
        "  --num-superstabilizers <int>           (default 0)\n"
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
        } else if (arg == "--d") {
            const char *v = need_val(arg.c_str());
            if (!parse_int(v, cfg.d)) { std::cerr << "Bad int: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "--r") {
            const char *v = need_val(arg.c_str());
            if (!parse_int(v, cfg.r)) { std::cerr << "Bad int: " << v << "\n"; return 2; }
            ++i;
        } else if (arg == "--num-superstabilizers") {
            const char *v = need_val(arg.c_str());
            if (!parse_int(v, cfg.num_superstabilizers)) { std::cerr << "Bad int: " << v << "\n"; return 2; }
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
