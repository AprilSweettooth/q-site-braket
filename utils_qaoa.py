# IMPORTS
import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.circuits.circuit import subroutine
from braket.devices import LocalSimulator
from braket.parametric import FreeParameter
from scipy.optimize import minimize
import networkx as nx
import time
import random
from braket.error_mitigation import Debias

from utils_classical import *

# function to implement ZZ gate using CNOT gates
@subroutine(register=True)
def ZZgate(q1, q2, gamma):
    """
    function that returns a circuit implementing exp(-i \gamma Z_i Z_j) using CNOT gates if ZZ not supported
    """
    # get a circuit
    circ_zz = Circuit()

    # construct decomposition of ZZ
    circ_zz.cnot(q1, q2).rz(q2, gamma).cnot(q1, q2)

    return circ_zz


# function to implement evolution with driver Hamiltonian
@subroutine(register=True)
def driver(beta, n_qubits):
    """
    Returns circuit for driver Hamiltonian U(Hb, beta)
    """
    # instantiate circuit object
    circ = Circuit()

    # apply parametrized rotation around x to every qubit
    for qubit in range(n_qubits):
        gate = Circuit().rx(qubit, 2 * beta)
        circ.add(gate)

    return circ


# helper function for evolution with cost Hamiltonian
@subroutine(register=True)
def cost_circuit(gamma, n_qubits, ising, device):
    """
    returns circuit for evolution with cost Hamiltonian
    """
    # instantiate circuit object
    circ = Circuit()

    # get all non-zero entries (edges) from Ising matrix
    idx = ising.nonzero()
    edges = list(zip(idx[0], idx[1]))

    # apply ZZ gate for every edge (with corresponding interaction strength)
    for qubit_pair in edges:
        # get interaction strength from Ising matrix
        int_strength = ising[qubit_pair[0], qubit_pair[1]]
        # for Rigetti we decompose ZZ using CNOT gates
        if isinstance(device, AwsDevice) and device.provider_name == "Rigetti":
            gate = ZZgate(qubit_pair[0], qubit_pair[1], gamma * int_strength)
            circ.add(gate)
        # classical simulators and IonQ support ZZ gate
        else:
            gate = Circuit().zz(qubit_pair[0], qubit_pair[1], angle=2 * gamma * int_strength)
            circ.add(gate)

    return circ


# function to build the QAOA circuit with depth p
def circuit(params, device, n_qubits, ising):
    """
    function to return full QAOA circuit; depends on device as ZZ implementation depends on gate set of backend
    """

    # initialize qaoa circuit with first Hadamard layer: for minimization start in |->
    circ = Circuit()
    X_on_all = Circuit().x(range(0, n_qubits))
    circ.add(X_on_all)
    H_on_all = Circuit().h(range(0, n_qubits))
    circ.add(H_on_all)

    # setup two parameter families
    circuit_length = int(len(params) / 2)
    gammas = params[:circuit_length]
    betas = params[circuit_length:]

    # add QAOA circuit layer blocks
    for mm in range(circuit_length):
        circ.cost_circuit(gammas[mm], n_qubits, ising, device)
        circ.driver(betas[mm], n_qubits)

    return circ

# function that computes cost function for given params
def objective_function(params, qaoa_circuit, ising, device, n_shots, tracker, verbose, G, error_mitigation):
    """
    objective function takes a list of variational parameters as input,
    and returns the cost associated with those parameters
    """

    if verbose:
        print("==================================" * 2)
        print("Calling the quantum circuit. Cycle:", tracker["count"])

    # create parameter dict
    params_dict = {str(fp): p for fp, p in zip(qaoa_circuit.parameters, params)}
    
    # classically simulate the circuit
    # set the parameter values using the inputs argument
    # execute the correct device.run call depending on whether the backend is local or cloud based
    if isinstance(device, LocalSimulator):
        task = device.run(qaoa_circuit, shots=n_shots, inputs=params_dict)
    else:
        if error_mitigation:
            task = device.run(qaoa_circuit, shots=n_shots, inputs=params_dict, poll_timeout_seconds=3 * 24 * 60 * 60, device_parameters={"errorMitigation": Debias()})
            
        else:
            task = device.run(
                qaoa_circuit, shots=n_shots, inputs=params_dict, poll_timeout_seconds=3 * 24 * 60 * 60
            )

    # get result for this task
    if error_mitigation:
        result = task.result()
        sharp_probs = result.additional_metadata.ionqMetadata.sharpenedProbabilities
    else:
        result = task.result()

    # convert results (0 and 1) to ising (-1 and 1)
    meas_ising = result.measurements
    # print(result.measurement_counts)
    # print(meas_ising.shape)
    meas_ising[meas_ising == 0] = -1

    # get all energies (for every shot): (n_shots, 1) vector
    all_energies = []
    # print(meas_ising)
    for i in range(meas_ising.shape[0]):
        all_energies.append(np.dot(meas_ising[i], np.dot(ising, np.transpose(meas_ising[i]))))

    # find minimum and corresponding classical string
    energy_min = np.min(all_energies)
    tracker["opt_energies"].append(energy_min)
    optimal_string = -meas_ising[np.argmin(all_energies)]
    tracker["opt_bitstrings"].append(optimal_string)
    # print(np.dot(ising, np.transpose(optimal_string)))
    # store optimal (classical) result/bitstring
    if energy_min < tracker["optimal_energy"]:
        tracker.update({"optimal_energy": energy_min})
        tracker.update({"optimal_bitstring": optimal_string})

    # store global minimum
    tracker["global_energies"].append(tracker["optimal_energy"])

    # energy expectation value
    energy_expect = np.sum(all_energies) / n_shots

    if verbose:
        print("Minimal energy:", energy_min)
        print("Optimal classical string:", optimal_string)
        print("Energy expectation value (cost):", energy_expect)
    # print('ratio:', energy_expect/energy_min)

    # update tracker
    tracker.update({"count": tracker["count"] + 1, "res": result})
    tracker["costs"].append(energy_expect)
    tracker["params"].append(params)
    global_min = list(solve_QAOA(G, ising).values())[0]
    tracker["ratio_min"].append(energy_min/global_min)
    tracker["ratio"].append(energy_expect/global_min)
    ops = ''
    for i in optimal_string:
        if i==-1:
            ops += str(0)
        else:
            ops += str(1)
    tracker['ratio_shot'].append(result.measurement_counts[''.join(list(str(i) for i in ops))]/sum(list(result.measurement_counts.values())))
    tracker['prob'].append(sharp_probs)
    return energy_expect


# The function to execute the training: run classical minimization.
def train(
    device, options, p, ising, n_qubits, n_shots, opt_method, tracker, verbose=True, parameter=None, G=None, error_mitigation=False
):
    """
    function to run QAOA algorithm for given, fixed circuit depth p
    """
    print("Starting the training.")

    print("==================================" * 2)
    print(f"OPTIMIZATION for circuit depth p={p}, print('Problem size:', {n_qubits}).")

    # if not verbose:
    #     print('Param "verbose" set to False. Will not print intermediate steps.')
    #     print("==================================" * 2)

    # initialize
    cost_energy = []

    # randomly initialize variational parameters within appropriate bounds
    if parameter is not None:
        circuit_length = int(len(parameter) / 2)
        gamma_initial_L = list(parameter[:circuit_length])
        gamma_initial_L.insert(0, 0)
        gamma_initial_L.insert(-1, 0)
        beta_initial_L = list(parameter[circuit_length:])
        beta_initial_L.insert(0, 0)
        beta_initial_L.insert(-1, 0)
        gamma_initial = []
        beta_initial = []
        # print(gamma_initial_L, (p-1)//2+2)
        for i in range(1,circuit_length+2):
            gamma_initial.append((i-1)*gamma_initial_L[i-1]/len(parameter)+(len(parameter)-i+1)*gamma_initial_L[i]/len(parameter))
        for i in range(1,circuit_length+2):
            beta_initial.append((i-1)*beta_initial_L[i-1]/len(parameter)+(len(parameter)-i+1)*beta_initial_L[i]/len(parameter))
        params0 = np.array(gamma_initial + beta_initial)
        # print(params0)
    else:
        gamma_initial = np.random.uniform(0, 2 * np.pi, p).tolist()
        beta_initial = np.random.uniform(0, np.pi, p).tolist()
        params0 = np.array(gamma_initial + beta_initial)
    

    # set bounds for search space
    bnds_gamma = [(0, 2 * np.pi) for _ in range(int(len(params0) / 2))]
    bnds_beta = [(0, np.pi) for _ in range(int(len(params0) / 2))]
    bnds = bnds_gamma + bnds_beta

    tracker["params"].append(params0)
    
    gamma_params = [FreeParameter(f"gamma_{i}") for i in range(p)]
    beta_params = [FreeParameter(f"beta_{i}") for i in range(p)]
    params = gamma_params + beta_params
    qaoa_circ = circuit(params, device, n_qubits, ising)

    # run classical optimization (example: method='Nelder-Mead')
    result = minimize(
        objective_function,
        params0,
        args=(qaoa_circ, ising, device, n_shots, tracker, verbose, G, error_mitigation),
        options=options,
        method=opt_method,
    )

    # store result of classical optimization
    result_energy = result.fun
    cost_energy.append(result_energy)
    print("Final average energy (cost):", result_energy)
    result_angle = result.x
    print("Final angles:", result_angle)
    print("Training complete.")

    return result_energy, result_angle, tracker

def benchmark(nodes_list, depth, shots=1000, bipartite=False, optimization=False, error_mitigation=False):
    seed = 42
    np.random.seed(seed)
    random.seed(a=seed)
    device = LocalSimulator()
    result = {}
    for n in nodes_list:
        data = {}
        for d in depth:
            n_data = {}
            if bipartite:
                n1 = np.random.randint(2,n)
                G = G = nx.bipartite.complete_bipartite_graph(n1,n-n1)
            else:
                G = nx.erdos_renyi_graph(n=n, p=0.6, seed=seed)
            # choose random weights
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = np.round(random.uniform(0, 1),4)
            # set Ising matrix 
            Jfull = nx.to_numpy_array(G)

            # get off-diagonal upper triangular matrix
            J = np.triu(Jfull, k=1).astype(np.float64)

            ##################################################################################
            # set up hyperparameters
            ##################################################################################

            # User-defined hypers
            DEPTH = d  # circuit depth for QAOA
            SHOTS = shots  # number measurements to make on circuit
            OPT_METHOD = 'COBYLA'  # SLSQP, COBYLA, Nelder-Mead, BFGS, Powell, ...

            # set up the problem
            n_qubits = J.shape[0]

            # initialize reference solution (simple guess)
            bitstring_init = -1 * np.ones([n_qubits])
            energy_init = np.dot(bitstring_init, np.dot(J, bitstring_init))

            # set tracker to keep track of results
            tracker = {
                'count': 1,                           # Elapsed optimization steps
                'optimal_energy': energy_init,        # Global optimal energy
                'opt_energies': [],                   # Optimal energy at each step
                'global_energies': [],                # Global optimal energy at each step
                'optimal_bitstring': bitstring_init,  # Global optimal bitstring
                'opt_bitstrings': [],                 # Optimal bitstring at each step
                'costs': [],                          # Cost (average energy) at each step
                'res': None,                          # Quantum result object
                'params': [],                         # Track parameters
                'ratio': [],                          # The benchmarking metric
                'ratio_min': [],                      # Best possible solution during measurement
                'ratio_shot': [],                     # The ratio for the optimal solution over all measurements
                'prob': []                             # Record prob of measurement
            }

            # set options for classical optimization
            options = {'disp': True, 'maxiter': 50}

            ##################################################################################
            # run QAOA optimization on graph 
            ##################################################################################

            # print('Circuit depth hyperparameter:', DEPTH)
            # print('Problem size:', n_qubits)

            # kick off training
            start = time.time()
            if DEPTH > depth[0] and optimization:
                result_energy, result_angle, tracker = train(
                    device = device, options=options, p=DEPTH, ising=J, n_qubits=n_qubits, n_shots=SHOTS, 
                    opt_method=OPT_METHOD, tracker=tracker, verbose=False, parameter=result_angle, G=G, error_mitigation=error_mitigation)
            else:
                result_energy, result_angle, tracker = train(
                    device = device, options=options, p=DEPTH, ising=J, n_qubits=n_qubits, n_shots=SHOTS, 
                    opt_method=OPT_METHOD, tracker=tracker, verbose=False, parameter=None, G=G, error_mitigation=error_mitigation)
            end = time.time()

            # print execution time
            # print('Code execution time [sec]:', end - start)

            # print optimized results
            # print('Optimal energy:', tracker['optimal_energy'])
            # print('Optimal classical bitstring:', tracker['optimal_bitstring'])
            # print("==================================" * 2)
            # print("==================================" * 2)
            # print("==================================" * 2)
            n_data['tracker'] = tracker
            n_data['energy'] = result_energy
            n_data['angle'] = result_angle
            n_data['time'] = end - start
            data['depth'+str(d)] = n_data
        result['node'+str(n)] = data

    return result


