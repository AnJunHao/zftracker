import networkx as nx
import numpy as np
from collections import Counter
from icecream import ic

def generate_full_graph(mapping):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph based on the mapping
    for src_group, targets in mapping.items():
        for target_group, ids in targets.items():
            for src_id, target_info in ids.items():
                if target_info is None:
                    source = (src_group, src_id)
                    G.add_node(source)
                    # Add attributes to the node
                    G.nodes[source]['center'] = False
                else:
                    # Create unique identifiers for entities by combining group name and id
                    source = (src_group, src_id)
                    target_id, distance = target_info
                    target = (target_group, target_id)
                    G.add_edge(source, target, distance=distance)
                    # Add attributes to the nodes
                    G.nodes[source]['center'] = False
                    G.nodes[target]['center'] = False
    return G

def get_subgraphs_from_mapping(mapping):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph based on the mapping
    for src_group, targets in mapping.items():
        for target_group, ids in targets.items():
            for src_id, target_info in ids.items():
                if target_info is None:
                    source = (src_group, src_id)
                    G.add_node(source)
                else:
                    # Create unique identifiers for entities by combining group name and id
                    source = (src_group, src_id)
                    target_id, distance = target_info
                    target = (target_group, target_id)
                    G.add_edge(source, target, distance=distance)

    # Get weakly connected components
    return get_weakly_connected_components(G)

def get_weakly_connected_components(graph):
    return [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]

def get_strongly_connected_components(graph):
    return [graph.subgraph(c) for c in nx.strongly_connected_components(graph)]

def get_score(graph, node, neighbor):
    score = 0
    if graph.has_edge(node, neighbor):
        distance = graph[node][neighbor]['distance']
        score += 1000
        score -= distance
    if graph.has_edge(neighbor, node):
        distance = graph[neighbor][node]['distance']
        score += 1000
        score -= distance
    return score

def digraph_get_neighbors(graph, node):
    successors = list(graph.successors(node))
    predecessors = list(graph.predecessors(node))
    return successors + predecessors

def single_node_connect_best_neighbors(graph, center_node):
    groups = set(('head', 'midsec', 'tail'))
    groups.remove(center_node[0])
    node_sets = {}
    first_neighbors = [node
                       for node in digraph_get_neighbors(graph, center_node)
                       if node[0] in groups]
    for first_neighbor in first_neighbors:
        score = get_score(graph, center_node, first_neighbor)
        node_sets[tuple(sorted([center_node, first_neighbor]))] = score
        remaining_group = groups - set([first_neighbor[0]])
        second_neighbors = [node
                            for node in digraph_get_neighbors(graph, first_neighbor)
                            if node[0] in remaining_group]
        for second_neighbor in second_neighbors:
            if tuple(sorted([center_node, first_neighbor, second_neighbor])) not in node_sets:
                score_1 = get_score(graph, first_neighbor, second_neighbor)
                score_2 = get_score(graph, center_node, second_neighbor)
                total_score = score + score_1 + score_2 + 1000 # 1000 is added to prioritize the number of nodes
                node_sets[tuple(sorted([center_node, first_neighbor, second_neighbor]))] = total_score
    best_nodes = max(node_sets, key=node_sets.get)
    return list(best_nodes)

def connected_nodes_select_best_neighbor(graph, center_nodes):
    groups = set(('head', 'midsec', 'tail'))
    groups.remove(center_nodes[0][0])
    groups.remove(center_nodes[1][0])
    # Since we have only three groups, there is only one group left
    # Every node in the remaining group must be a neighbor of the center nodes
    neighbors = [node for node in graph.nodes if node[0] in groups]
    scores = []
    for node in neighbors:
        scores.append(get_score(graph, center_nodes[0], node) +
                      get_score(graph, center_nodes[1], node))
    best_neighbor = neighbors[np.argmax(scores)]
    return list(center_nodes) + [best_neighbor]

def disconnected_nodes_select_best_neighbor(graph, center_nodes):
    groups = set(('head', 'midsec', 'tail'))
    groups.remove(center_nodes[0][0])
    groups.remove(center_nodes[1][0])
    # Since we have only three groups, there is only one group left
    # Every node in the remaining group must be a neighbor of the center nodes
    # The center nodes are not connected to each other, but they stay within the same graph
    # This means that there must be at least one node in the remaining group that is connected to both center nodes
    neighbors = [node
                 for node in graph.nodes
                 if (node[0] in groups and
                    (graph.has_edge(center_nodes[0], node) or graph.has_edge(node, center_nodes[0])) and
                    (graph.has_edge(center_nodes[1], node) or graph.has_edge(node, center_nodes[1]))
                    )]
    if len(neighbors) == 1:
        return list(center_nodes) + neighbors
    else: # len(neighbors) > 1 (len(neighbors) == 0 is not possible)
        scores = []
        for node in neighbors:
            scores.append(get_score(graph, center_nodes[0], node) +
                          get_score(graph, center_nodes[1], node))
        best_neighbor = neighbors[np.argmax(scores)]
        return list(center_nodes) + [best_neighbor]
    
def get_repeated_and_not_repeated_groups(graph):
    seen_groups = set()
    repeated_groups = set()
    for node in graph.nodes:
        if node[0] in seen_groups:
            repeated_groups.add(node[0])
        seen_groups.add(node[0])
    not_repeated_groups = seen_groups - repeated_groups
    return list(repeated_groups), list(not_repeated_groups)

def remove_edges_between_nodes(graph, node_a, node_b):
    if graph.has_edge(node_a, node_b):
        graph.remove_edge(node_a, node_b)
    if graph.has_edge(node_b, node_a):
        graph.remove_edge(node_b, node_a)
    subgraphs = get_weakly_connected_components(graph)
    return subgraphs
    
def divide_graph(graph):
    # Given a graph with at least 3 nodes, we attempt to divide the graph into two subgraphs
    # The division is performed on the center code (not repeated group) of the graph
    repeated_groups, not_repeated_groups = get_repeated_and_not_repeated_groups(graph)
    if len(not_repeated_groups) > 0:
        center_nodes = [node for node in graph.nodes if node[0] in not_repeated_groups]
        if len(center_nodes) == 2:
            # If there are two center nodes, we have to check if they are connected
            if not (graph.has_edge(center_nodes[0], center_nodes[1]) or
                    graph.has_edge(center_nodes[1], center_nodes[0])):
                # If the two center nodes are not connected, we cannot divide the graph
                return False, []
        graph_copy = graph.copy()
        graph_copy.remove_nodes_from(center_nodes)
        weakly_connected_subgraphs = get_weakly_connected_components(graph_copy)
        if len(weakly_connected_subgraphs) == 2:
            # We then check if the two subgraphs have no repeated groups
            for subgraph in weakly_connected_subgraphs:
                repeated_groups, not_repeated_groups = get_repeated_and_not_repeated_groups(subgraph)
                if len(repeated_groups) > 0:
                    # If there is a repeated group in the subgraph, we cannot divide the graph
                    return False, []
            # We then add the center nodes to the two subgraphs
            new_subgraphs = []
            for subgraph in weakly_connected_subgraphs:
                subgraph_nodes = list(subgraph.nodes)
                # Ensure that the center nodes are connected to the subgraph
                connected_center_nodes = []
                for center_node in center_nodes:
                    neighbors = digraph_get_neighbors(graph, center_node)
                    if any([node in neighbors for node in subgraph_nodes]):
                        connected_center_nodes.append(center_node)
                new_subgraph = graph.subgraph(subgraph_nodes + connected_center_nodes)
                # Modify attributes of the center nodes
                for node in connected_center_nodes:
                    new_subgraph.nodes[node]['center'] = True
                new_subgraphs.append(new_subgraph)
            return True, new_subgraphs
        else:
            # If the number of subgraphs is not 2, we cannot divide the graph
            return False, []
    else: # len(not_repeated_groups) == 0
        # In this case, there is no center node for reference
        # We attempt to divde the graph into two when the number of nodes is 4 or 6
        if len(graph.nodes) == 4 and len(repeated_groups) == 2:
            # The graph must look like head-tail-head-tail (the first and fourth node are not connected)
            # In this case, we can divide the graph between the second and third nodes,
            # so that the subgraphs are head-tail and head-tail
            nodes = list(graph.nodes)
            node_neighbor_count = [digraph_get_neighbors(graph, node) for node in nodes]
            counter = Counter(node_neighbor_count)
            if counter[1] == 2 and counter[2] == 2: # The first and fourth node have 1 neighbor, the second and third node have 2 neighbors
                 connecting_nodes = [nodes[i] for i, count in enumerate(node_neighbor_count) if count == 2]
                 subgraphs = remove_edges_between_nodes(graph, connecting_nodes[0], connecting_nodes[1])
                 return True, subgraphs
        elif len(graph.nodes) == 6 and len(repeated_groups) == 3:
            # The graph must look like (head-midsec-tail)-(head-midsec-tail)
            # The first three nodes may be connected to each other, so may the last three
            # However, there must be one and only one connection between the first three nodes and the last three nodes
            # TO BE IMPLEMENTED
            return False, []
        else:
            return False, []

def analyze_full_graph_v3(graph, num_entities):

    # Get weakly connected components
    weakly_connected_subgraphs = get_weakly_connected_components(graph)

    graph_pools = {'original': [], 'denoised': [], 'divided': []}
    awaiting_graph_pool = []
    for subgraph in weakly_connected_subgraphs:
        seen_groups = set()
        repeated_groups = set()
        for node in subgraph.nodes:
            if node[0] not in seen_groups:
                seen_groups.add(node[0])
            else:
                repeated_groups.add(node[0])
        if len(repeated_groups) == 0 and 2 <= len(seen_groups) <= 3:
            graph_pools['original'].append(subgraph)
        else:
            awaiting_graph_pool.append(subgraph)

    if len(graph_pools['original']) >= num_entities:
        graph_pools['original'].sort(key=lambda x: (len(x.nodes), len(x.edges)), reverse=True)
        graph_pools['original'] = graph_pools['original'][:num_entities]
        return graph_pools, []
    
    awaiting_graph_pool.sort(key=lambda x: (len(x.nodes), len(x.edges)), reverse=True) # in descending order
    unprocessed_graph_pool = []

    while (num_entities - len(graph_pools['original']) - len(graph_pools['denoised']) - len(graph_pools['divided']) > 0 and
           len(awaiting_graph_pool) > 0):

        missing_entities = num_entities - len(graph_pools['original']) - len(graph_pools['denoised']) - len(graph_pools['divided'])
        candidate_graph = awaiting_graph_pool.pop(0) # get the graph with the largest number of nodes & edges

        if len(candidate_graph.nodes) == 1:
            if len(unprocessed_graph_pool) > 0:
                # If there are unprocessed graphs, we add the current graph to the unprocessed graph pool
                unprocessed_graph_pool.append(candidate_graph)
            else:
                # There are no unprocessed graphs, the remaining awaiting graphs are all single nodes
                num_awaiting_nodes = len(awaiting_graph_pool) + 1 # including the current graph
                if missing_entities >= num_awaiting_nodes:
                    graph_pools['original'].append(candidate_graph)
                    graph_pools['original'] += awaiting_graph_pool
                    break
                else: # The number of awaiting nodes is greater than the missing entities
                    # We can't determine which node to add to the active graph pool
                    unprocessed_graph_pool.append(candidate_graph)
                    unprocessed_graph_pool += awaiting_graph_pool
                    break
        elif len(candidate_graph.nodes) >= 3 and missing_entities == 1: # candidate_graph must have 1 nodes or >= 3 nodes
            # In this case, we need to eliminate one of the repeated nodes
            # Find the center node of the graph (center node are non-repeated nodes)
            seen_groups = set()
            repeated_groups = set()
            for node in candidate_graph.nodes:
                if node[0] in seen_groups:
                    repeated_groups.add(node[0])
                seen_groups.add(node[0])
            not_repeated_groups = list(seen_groups - repeated_groups)
            if len(not_repeated_groups) == 1:
                center_node = [node for node in candidate_graph.nodes if node[0] == not_repeated_groups[0]][0]
                best_components = single_node_connect_best_neighbors(candidate_graph, center_node)
                extracted_graph = candidate_graph.subgraph(best_components)
                graph_pools['denoised'].append(extracted_graph)
            elif len(not_repeated_groups) == 2:
                center_nodes = [node for node in candidate_graph.nodes if node[0] in not_repeated_groups]
                if (candidate_graph.has_edge(center_nodes[0], center_nodes[1]) or
                    candidate_graph.has_edge(center_nodes[1], center_nodes[0])):
                    best_components = connected_nodes_select_best_neighbor(candidate_graph, center_nodes)
                    extracted_graph = candidate_graph.subgraph(best_components)
                    graph_pools['denoised'].append(extracted_graph)
                else:
                    best_components = disconnected_nodes_select_best_neighbor(candidate_graph, center_nodes)
                    extracted_graph = candidate_graph.subgraph(best_components)
                    graph_pools['denoised'].append(extracted_graph)
            else:
                print('Warning: More than two non-repeated groups found in the graph.')
        elif len(candidate_graph.nodes) >= 3 and missing_entities >= 2:
            if len(awaiting_graph_pool) + 1 >= missing_entities:
                # In this case, the current possible new entities are enough to fill the missing entities
                # We cannot determine whether to divide the graph or not
                unprocessed_graph_pool.append(candidate_graph)
                unprocessed_graph_pool += awaiting_graph_pool
                break
            # In this case, we divde the graph into two subgraphs
            is_success, subgraphs = divide_graph(candidate_graph)
            if is_success:
                for subgraph in subgraphs:
                    graph_pools['divided'].append(subgraph)
            else:
                unprocessed_graph_pool.append(candidate_graph)
        else:
            raise ValueError('Analyzation entered an unexpected state.')
    return graph_pools, unprocessed_graph_pool

def analyze_full_graph_v2(graph, num_entities):
    # Get weakly connected components
    weakly_connected_subgraphs = get_weakly_connected_components(graph)
    active_graph_pool = []
    awaiting_graph_pool = []
    for subgraph in weakly_connected_subgraphs:
        seen_groups = set()
        repeated_groups = set()
        for node in subgraph.nodes:
            if node[0] not in seen_groups:
                seen_groups.add(node[0])
            else:
                repeated_groups.add(node[0])
        if len(repeated_groups) == 0 and 2 <= len(seen_groups) <= 3:
            active_graph_pool.append(subgraph)
        else:
            awaiting_graph_pool.append(subgraph)
    missing_entities = num_entities - len(active_graph_pool)
    if missing_entities == 0:
        return active_graph_pool, [], 0
    elif missing_entities < 0:
        # sort the active graph pool by the number of nodes and then by the number of edges
        active_graph_pool.sort(key=lambda x: (len(x.nodes), len(x.edges)), reverse=True) # in descending order
        return active_graph_pool[:num_entities], [], 0
    elif len(awaiting_graph_pool) == missing_entities: # missing_entities > 0
        if all([len(subgraph.nodes) == 1 for subgraph in awaiting_graph_pool]):
            active_graph_pool += awaiting_graph_pool
            return active_graph_pool, [], 0
        else:
            return active_graph_pool, awaiting_graph_pool, missing_entities
    elif len(awaiting_graph_pool) < missing_entities: # missing_entities > 0
        if all([len(subgraph.nodes) == 1 for subgraph in awaiting_graph_pool]):
            active_graph_pool += awaiting_graph_pool
            return active_graph_pool, [], 0
        else:
            return active_graph_pool, awaiting_graph_pool, missing_entities
    else: # len(awaiting_graph_pool) > missing_entities and missing_entities > 0
        return active_graph_pool, awaiting_graph_pool, missing_entities
    
def analyze_full_graph(graph):
    # Get weakly connected components
    weakly_connected_subgraphs = get_weakly_connected_components(graph)
    graph_pool = [{'n_nodes': None,
                        'n_edges': None,
                        'graph': subgraph,
                        'active': False,
                        'original': True}
                    for subgraph in weakly_connected_subgraphs]
    i = -1
    while i < len(graph_pool) - 1:
        i += 1
        subgraph = graph_pool[i]
        if subgraph['active']:
            continue
        # See if there is a head-midsec-tail group in the graph
        seen_groups = set()
        repeated_groups = set()
        for node in subgraph['graph'].nodes:
            if node[0] not in seen_groups:
                seen_groups.add(node[0])
            else:
                repeated_groups.add(node[0])
        if len(repeated_groups) == 0: # Nothing to do if there is no repeated group
            subgraph['active'] = True
            subgraph['n_nodes'] = len(seen_groups)
            subgraph['n_edges'] = len(subgraph['graph'].edges)
            continue
        not_repeated_groups = list(seen_groups - repeated_groups)
        if len(not_repeated_groups) == 1:
            # Find the node with the non-repeated group
            for center_node in subgraph['graph'].nodes:
                if center_node[0] == not_repeated_groups[0]:
                    break
            best_components = single_node_connect_best_neighbors(subgraph['graph'], center_node)
            # Extract the subgraph with the center node and the best neighbors
            extracted_graph = subgraph['graph'].subgraph(best_components)
            graph_pool.append({'n_nodes': len(extracted_graph.nodes),
                                'n_edges': len(extracted_graph.edges),
                                'graph': extracted_graph,
                                'active': True,
                                'original': False})
            # Remove the extracted graph from the original graph
            graph_copy = subgraph['graph'].copy()
            graph_copy.remove_nodes_from(extracted_graph.nodes)
            weakly_connected_subgraphs = get_weakly_connected_components(graph_copy)
            for subgraph in weakly_connected_subgraphs:
                graph_pool.append({'n_nodes': None,
                                    'n_edges': None,
                                    'graph': subgraph,
                                    'active': False,
                                    'original': False})
        elif len(not_repeated_groups) == 2:
            # Find the nodes with the non-repeated groups
            center_nodes = []
            for node in subgraph['graph'].nodes:
                if node[0] in not_repeated_groups:
                    center_nodes.append(node)
            # Check if the two nodes are connected
            if (subgraph['graph'].has_edge(center_nodes[0], center_nodes[1]) or
                subgraph['graph'].has_edge(center_nodes[1], center_nodes[0])):
                best_components = connected_nodes_select_best_neighbor(subgraph['graph'], center_nodes)
                # Extract the subgraph with the center nodes and the best neighbors
                extracted_graph = subgraph['graph'].subgraph(best_components)
                graph_pool.append({'n_nodes': len(extracted_graph.nodes),
                                    'n_edges': len(extracted_graph.edges),
                                    'graph': extracted_graph,
                                    'active': True,
                                    'original': False})
                # Remove the extracted graph from the original graph
                graph_copy = subgraph['graph'].copy()
                graph_copy.remove_nodes_from(extracted_graph.nodes)
                weakly_connected_subgraphs = get_weakly_connected_components(graph_copy)
                for subgraph in weakly_connected_subgraphs:
                    graph_pool.append({'n_nodes': None,
                                        'n_edges': None,
                                        'graph': subgraph,
                                        'active': False,
                                        'original': False})
            else:
                # The two nodes are not connected
                best_components = disconnected_nodes_select_best_neighbor(subgraph['graph'], center_nodes)
                # Extract the subgraph with the center nodes and the best neighbors
                extracted_graph = subgraph['graph'].subgraph(best_components)
                graph_pool.append({'n_nodes': len(extracted_graph.nodes),
                                    'n_edges': len(extracted_graph.edges),
                                    'graph': extracted_graph,
                                    'active': True,
                                    'original': False})
                # Remove the extracted graph from the original graph
                graph_copy = subgraph['graph'].copy()
                graph_copy.remove_nodes_from(extracted_graph.nodes)
                weakly_connected_subgraphs = get_weakly_connected_components(graph_copy)
                for subgraph in weakly_connected_subgraphs:
                    graph_pool.append({'n_nodes': None,
                                        'n_edges': None,
                                        'graph': subgraph,
                                        'active': False,
                                        'original': False})
        elif len(not_repeated_groups) == 0:
            # Here, we select center nodes based on the degree of the nodes
            nodes = []
            degrees = []
            for node, degree in subgraph['graph'].degree:
                nodes.append(node)
                degrees.append(degree)
            center_node = nodes[np.argmax(degrees)]
            best_components = single_node_connect_best_neighbors(subgraph['graph'], center_node)
            # Extract the subgraph with the center node and the best neighbors
            extracted_graph = subgraph['graph'].subgraph(best_components)
            graph_pool.append({'n_nodes': len(extracted_graph.nodes),
                                'n_edges': len(extracted_graph.edges),
                                'graph': extracted_graph,
                                'active': True,
                                'original': False})
            # Remove the extracted graph from the original graph
            graph_copy = subgraph['graph'].copy()
            graph_copy.remove_nodes_from(extracted_graph.nodes)
            weakly_connected_subgraphs = get_weakly_connected_components(graph_copy)
            for subgraph in weakly_connected_subgraphs:
                graph_pool.append({'n_nodes': None,
                                    'n_edges': None,
                                    'graph': subgraph,
                                    'active': False,
                                    'original': False})
        else:
            print('Warning: More than two non-repeated groups found in the graph.')
    return graph_pool