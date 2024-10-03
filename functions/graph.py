import torch
import networkx as nx

def getGraph(station):
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    num_nodes = 5
    G.add_nodes_from(range(num_nodes))
    
    Varberg_UFC, OKQ8_Svansjögatan, Jureskogs_Vattenfall = 0,0,0
    IONITY_Varberg, Emporia_plan_3, IONITY = 1,1,1
    Toveks_Bil, P_Huset_Plan_2, UFC = 2,2,2
    Varberg_Supercharger, OKQ8_Kvartettgatan, Holmgrens = 3,3,3
    Recharge_ST1, P_hus_Malmömässan, donken = 4,4,4
    
    if station == "varnamo":
        edges_to_connect = [
            (Jureskogs_Vattenfall, IONITY, 230),
            (Jureskogs_Vattenfall, UFC, 750),
            (Jureskogs_Vattenfall, Holmgrens, 750),
            (Jureskogs_Vattenfall, donken, 650),
            (IONITY, UFC, 550),
            (IONITY, Holmgrens, 500),
            (IONITY, donken, 450),
            (UFC, Holmgrens, 280),
            (UFC, donken, 550),
            (Holmgrens, donken, 550)
        ]
    elif station == "varberg":
        edges_to_connect = [
            (Varberg_UFC, IONITY_Varberg, 120),
            (Varberg_UFC, Toveks_Bil, 750),
            (Varberg_UFC, Varberg_Supercharger, 290),
            (Varberg_UFC, Recharge_ST1, 10900),
            (IONITY_Varberg, Toveks_Bil, 650),
            (IONITY_Varberg, Varberg_Supercharger, 270),
            (IONITY_Varberg, Recharge_ST1, 10800),
            (Toveks_Bil, Varberg_Supercharger, 650),
            (Toveks_Bil, Recharge_ST1, 11200),
            (Varberg_Supercharger, Recharge_ST1, 11000)
        ]
    elif station == "malmo":
        edges_to_connect = [
            (OKQ8_Svansjögatan, Emporia_plan_3 , 2300),
            (OKQ8_Svansjögatan, P_Huset_Plan_2 , 2800 ),
            (OKQ8_Svansjögatan, OKQ8_Kvartettgatan, 4900),
            (OKQ8_Svansjögatan, P_hus_Malmömässan, 2700),
            (Emporia_plan_3, P_Huset_Plan_2, 650),
            (Emporia_plan_3, OKQ8_Kvartettgatan, 2600),
            (Emporia_plan_3, P_hus_Malmömässan, 600),
            (P_Huset_Plan_2, OKQ8_Kvartettgatan, 3000),
            (P_Huset_Plan_2, P_hus_Malmömässan, 800),
            (OKQ8_Kvartettgatan, P_hus_Malmömässan, 3300)
        ]
    else:
        print(station, "not yet implimented")
        
    # Add edges with custom weights
    for edge in edges_to_connect:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    # Get edge index and edge attributes
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_attr = torch.tensor([d['weight'] for u, v, d in G.edges(data=True)])
    
    # Define colors for nodes
    #node_colors = ['Yellow'] * num_nodes
    
    #print(G.nodes)
    
    # Draw the graph
    #pos = nx.spring_layout(G)  # positions for all nodes
    #nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10)
    #edge_labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Show the graph
    #plt.show()

    return (edge_index, edge_attr)

if __name__ == "__main__":
    (edge_index, edge_attr) = getGraph("varnamo")
    print(edge_index.shape)
    print(edge_attr.shape)
