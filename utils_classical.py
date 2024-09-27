# IMPORTS
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def decimal_to_binary(decimal,num_nodes):
    binary_num = []
    # Outputs bitstring of 1s and 0s into an array of digits
    def convert(decimal):
        if decimal >= 1:
            convert(decimal // 2)
            binary_num.append(decimal % 2)
    
    convert(decimal)
            
    # Change the binary number to have 4 digits, if it doesn't already
    for i in range(num_nodes + 1):
        if len(binary_num) < i:
            binary_num.insert(0, 0) # At beginning append 0
    
    return binary_num # Outputs array of the digits of the binary number

def solve_QAOA(G,J):
    num_nodes = list(G.nodes)[-1]+1
    sol = {}
    for bitstring in range(2 ** num_nodes): # For each bitstring, we will calculate the cost classically without the quantum circuit.
        binary_bit_string = decimal_to_binary(bitstring, num_nodes) # Array type
        # print(binary_bit_string)
        bit_string_cost = 0
        # for edge in list(G.edges):
        # Jfull = nx.to_numpy_array(G)

        # get off-diagonal upper triangular matrix
        # J = np.triu(Jfull, k=1).astype(np.float64)
        temp = binary_bit_string.copy()
        for idx, t in enumerate(temp):
            if t == 0:
                temp[idx] -= 1
        # print(temp)
        # print('here',np.dot(np.array(binary_bit_string), np.dot(J, np.transpose(np.array(binary_bit_string)))))
        energy_min = np.dot(np.array(temp), np.dot(J, np.transpose(np.array(temp))))
        # print(np.dot(J, np.transpose(np.array(temp))))
        # print(J)

        # find minimum and corresponding classical string
        # energy_min = np.min(all_energies)
        bit_string_cost += energy_min
            # start_node = edge[0]
            # end_node = edge[1]
            # weight = G.edges[(edge)]['weight']
            # weighted_cost = -1 * (weight * binary_bit_string[start_node] * (1 - binary_bit_string[end_node]) + weight * binary_bit_string[end_node] * (1 - binary_bit_string[start_node])) 
            # weighted_cost = -1 * weight * (1 - binary_bit_string[start_node] * binary_bit_string[end_node])
            # bit_string_cost += weighted_cost
        
        binary_bit_string_2 = ''
        for bit in binary_bit_string: # Gets the string version of the array type binary_bit_string
                binary_bit_string_2 += str(bit)
        sol[binary_bit_string_2] = bit_string_cost
        # print(f'Cost of {binary_bit_string_2}: {bit_string_cost}')
    
    min_cost = min(list(sol.values()))
    for k, v in zip(list(sol.keys()), list(sol.values())):
        if v != min_cost:
            del sol[k]

    return sol

# helper function to plot graph
def plot_colored_graph_simple(graph, colors, pos):
    """
    plot colored graph for given colored solution
    """

    # define color scheme
    colorlist = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
    ]

    # draw network
    nx.draw_networkx(
        graph,
        pos,
        node_color=[colorlist[colors[int(node)]] for node in graph.nodes],
        node_size=400,
        font_weight="bold",
        font_color="w",
    )

    # plot the graph
    plt.axis("off")
    # plt.savefig("./figures/weighted_graph.png") # save as png
    # plt.show();


# helper function to plot graph
def plot_colored_graph(J, N, colors, pos):
    """
    plot colored graph for given colored solution
    """
    # define graph
    graph = nx.Graph()
    all_weights = []

    for ii in range(0, N):
        for jj in range(ii + 1, N):
            if J[ii][jj] != 0:
                graph.add_edge(str(ii), str(jj), weight=J[ii][jj])
                all_weights.append(J[ii][jj])

    # positions for all nodes
    # pos = nx.spring_layout(graph)

    # get unique weights
    unique_weights = list(set(all_weights))

    # plot the edges - one by one
    for weight in unique_weights:
        # form a filtered list with just the weight you want to draw
        weighted_edges = [
            (node1, node2)
            for (node1, node2, edge_attr) in graph.edges(data=True)
            if edge_attr["weight"] == weight
        ]
        # multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        # width = weight
        width = weight * N * 5.0 / sum(all_weights)
        nx.draw_networkx_edges(graph, pos, edgelist=weighted_edges, width=width)

    colorlist = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
    ]
    nx.draw_networkx(
        graph,
        pos,
        node_color=[colorlist[colors[int(node)]] for node in graph.nodes],
        node_size=400,
        font_weight="bold",
        font_color="w",
    )

    # plot the graph
    plt.axis("off")
    # plt.savefig("./figures/weighted_graph.png") # save as png
    # plt.show();