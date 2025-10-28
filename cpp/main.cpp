#include <stim.h>                  // or the correct include path for your Stim C++ API
#include "LogicalQubit.cpp"        // your class header
#include <fstream>
#include <iostream>

int main() {
    try {
        int d = 5;  // code distance
        double readout_err = 0.001;
        double gate1_err = 0.0005;
        double gate2_err = 0.005;
        int rounds = 10;

        // Example: no missing coordinates
        std::vector<std::pair<int,int>> missing_coords;

        // Construct logical qubit
        LogicalQubit logi(d, readout_err, gate1_err, gate2_err, missing_coords, 0.25, /*verbose=*/true);

        // Generate Stim circuit
        stim::Circuit circuit = logi.generate_stim(rounds);

        // Save to file
        std::ofstream out("logical_qubit_circuit.stim");
        if (!out) {
            std::cerr << "Failed to open output file.\n";
            return 1;
        }
        out << circuit.str();  // circuit.str() returns the full Stim text form
        out.close();

        std::cout << "✅ Saved circuit to logical_qubit_circuit.stim (" 
                  << rounds << " rounds, d=" << d << ")\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
