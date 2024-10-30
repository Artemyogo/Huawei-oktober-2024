import matplotlib.pyplot as plt
import networkx as nx


def parse_input(filename):
    with open(filename, 'r') as file:
        # Считываем количество узлов
        n = int(file.readline().strip())
        nodes = {}

        # Считываем узлы
        for _ in range(n):
            node_data = file.readline().strip().split()
            if len(node_data) != 3:
                raise ValueError(f"Node data format error: expected 3 values, got {len(node_data)}")
            node_id = int(node_data[0])
            lat, lon = float(node_data[1]), float(node_data[2])
            nodes[node_id] = (lat, lon)

        # Считываем количество рёбер
        m = int(file.readline().strip())
        edges = []

        # Считываем рёбра
        for _ in range(m):
            edge_data = file.readline().strip().split()
            if len(edge_data) != 2:
                raise ValueError(f"Edge data format error: expected 2 values, got {len(edge_data)}")
            u, v = map(int, edge_data)
            edges.append((u, v))

        # Считываем количество пользователей
        t = int(file.readline().strip())
        users = []

        # Считываем пользователей
        for _ in range(t):
            user_data = file.readline().strip().split()
            if len(user_data) != 3:
                raise ValueError(f"User data format error: expected 3 values, got {len(user_data)}")
            user_id = int(user_data[0])
            user_lat, user_lon = float(user_data[1]), float(user_data[2])
            users.append((user_id, user_lat, user_lon))

    return nodes, edges, users

def create_graph(nodes, edges):
    # Создание графа с использованием networkx
    G = nx.Graph()
    for node_id, coords in nodes.items():
        G.add_node(node_id, pos=coords)
    G.add_edges_from(edges)
    return G

def draw_graph(G, users):
    # Отрисовка графа и позиций пользователей
    plt.figure(figsize=(12, 8))

    # Позиции узлов
    pos = nx.get_node_attributes(G, 'pos')

    # Отрисовка узлов
    nx.draw_networkx_nodes(G, pos, node_size=2, node_color="blue", alpha=0.6, label="Nodes")

    # Отрисовка рёбер
    nx.draw_networkx_edges(G, pos, width=1, edge_color="gray", alpha=0.3)

    # Отрисовка пользователей, если они есть
    user_lat_lon = [(user[1], user[2]) for user in users]
    if user_lat_lon:
        user_x, user_y = zip(*user_lat_lon)
        plt.scatter(user_x, user_y, c="red", s=1, alpha=0.7, label="Users")
    else:
        print("Warning: No user data to display.")

    # Заголовок и легенда
    plt.title("Graph with Users")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()

def main():
    # Указываем имя файла с входными данными
    filename = 'graph_data.txt'
    nodes, edges, users = parse_input(filename)

    # Проверка корректности данных перед построением графа
    if not nodes or not edges or not users:
        print("Error: One or more data structures (nodes, edges, users) are empty or not parsed correctly.")
        return

    G = create_graph(nodes, edges)
    
    # Проверка данных пользователей перед их отображением
    if users:
        draw_graph(G, users)
    else:
        print("Error: Users data is empty, cannot proceed with drawing the graph.")

# Запуск главной функции
main()