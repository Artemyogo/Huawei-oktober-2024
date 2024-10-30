import tkinter as tk
import networkx as nx
from PIL import Image, ImageDraw
from decimal import Decimal

# Initialize the graph and variables for handling edges
G = nx.Graph()
user_id_counter = 1
node_id_counter = 1
positions = {}
first_node_selected = None  # Track the first node for edge creation

# Create a Tkinter window
root = tk.Tk()
root.title("Graph Drawing Tool")

# Create a canvas for drawing
canvas = tk.Canvas(root, width=800, height=600, bg='white')
canvas.pack()

def refresh_graph():
    canvas.delete("all")  # Clear the canvas
    # Draw nodes
    for node, (x, y) in positions.items():
        # Draw circle for each node
        radius = 10 if 'User' not in node else 15  # Different sizes for users
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="skyblue", outline="black")
        canvas.create_text(x, y, text=node, font=("Arial", 10))
    
    # Draw edges
    for u, v in G.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        canvas.create_line(x0, y0, x1, y1, fill="gray", width=2)

# Function to handle mouse clicks
def on_click(event):
    global user_id_counter, node_id_counter, first_node_selected
    
    # Right-click (button=3) to add a user node
    if event.num == 3:  # Right-click
        user_id = f"User{user_id_counter}"
        user_id_counter += 1
        G.add_node(user_id, pos=(event.x, event.y))
        positions[user_id] = (event.x, event.y)
        print(f"Added user node '{user_id}' at ({event.x}, {event.y})")
        refresh_graph()
    
    # Left-click (button=1) to add a regular graph node (vertex)
    elif event.num == 1:  # Left-click
        node_id = f"Node{node_id_counter}"
        node_id_counter += 1
        G.add_node(node_id, pos=(event.x, event.y))
        positions[node_id] = (event.x, event.y)
        print(f"Added graph node '{node_id}' at ({event.x}, {event.y})")
        refresh_graph()

    # Middle-click (button=2) to add an edge between two nodes
    elif event.num == 2:  # Middle-click
        # Check if there's an existing node near the click position
        selected_node = None
        for node, (x, y) in positions.items():
            if abs(x - event.x) < 15 and abs(y - event.y) < 15:  # Adjust tolerance for node selection
                selected_node = node
                break

        if selected_node is None:
            print("No node near this click position.")
            return

        if first_node_selected is None:
            first_node_selected = selected_node
            print(f"Selected first node '{first_node_selected}' for edge creation.")
        elif first_node_selected == selected_node:
            print("Selected the same node")
        else:
            G.add_edge(first_node_selected, selected_node)
            print(f"Connected '{first_node_selected}' to '{selected_node}' with an edge.")
            first_node_selected = None  # Reset for the next edge
            refresh_graph()

# Bind mouse click events to the canvas
canvas.bind("<Button-1>", on_click)  # Left click
canvas.bind("<Button-2>", on_click)  # Middle click
canvas.bind("<Button-3>", on_click)  # Right click

# Function to save the current graph as an image
def save_graph_image():
    # Create an image of the current graph (requires Pillow)
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    
    # Draw nodes and edges on the image
    for u, v in G.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        draw.line((x0, y0, x1, y1), fill="gray", width=2)
        
    for node, (x, y) in positions.items():
        radius = 10 if 'User' not in node else 15  # Different sizes for users
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="skyblue", outline="black")
        draw.text((x, y), node, fill="black")  # Add text to nodes
    
    img.save("graph_output.png")
    print("Graph saved as graph_output.png")

# Function to save the graph data to a text file
def save_graph_data(filename="graph_data.txt"):
    with open(filename, 'w') as f:
        # Write number of nodes
        num_users = user_id_counter - 1
        num_nodes = len(G.nodes()) - num_users
        f.write(f"{num_nodes}\n")
        curdel = 10

        # Write nodes and their positions
        for i in range(1, num_nodes + 1):
            node = f"Node{i}"
            lat = positions[node][0]  # Using x position as latitude
            lon = positions[node][1]  # Using y position as longitude
            f.write(f"{i} {Decimal(lat)/curdel:.15f} {Decimal(lon)/curdel:.15f}\n")
        # for node in G.nodes():
        #     lat = positions[node][0]  # Using x position as latitude
        #     lon = positions[node][1]  # Using y position as longitude
        #     f.write(f"{node} {lat:.15f} {lon:.15f}\n")

        # Write number of edges
        num_edges = len(G.edges())
        f.write(f"{num_edges}\n")

        # Write edges
        for u, v in G.edges():
            f.write(f"{u[4:]} {v[4:]}\n")

        # Write number of users
        f.write(f"{num_users}\n")

        # Write user nodes and their positions
        for i in range(1, num_users + 1):
            user_node = f"User{i}"
            lat = positions[user_node][0]  # Using x position as latitude
            lon = positions[user_node][1]  # Using y position as longitude
            f.write(f"{i} {Decimal(lat)/curdel:.15f} {Decimal(lon)/curdel:.15f}\n")

    print(f"Graph data saved as {filename}")

# Add a save button for graph data
save_button = tk.Button(root, text="Save Graph Data", command=save_graph_data)
save_button.pack()

# Add a save button for the graph image
save_image_button = tk.Button(root, text="Save Graph Image", command=save_graph_image)
save_image_button.pack()

# Run the Tkinter main loop
root.mainloop()