from dataclasses import dataclass
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import random
import matplotlib.patches as mpatches
from collections import deque

@dataclass(frozen=True)
class Node:
    x: int
    y: int
    blocked: bool = False
    cost: float = 1.0  # Cost to enter this node

    def __lt__(self, other):
        return False  # Required for heapq to compare Node instances

# Manhattan distance for heuristic
def h(n1: Node, n2: Node):
    return abs(n1.x - n2.x) + abs(n1.y - n2.y)

# Get neighbors of a node
def get_neighbors(node: Node, grid):
    neighbors = []
    directions = [(-1,0),(1,0),(0,-1),(0,1)]  # Up, Down, Left, Right
    x, y = node.x, node.y
    height = len(grid)
    width = len(grid[0])
    for dx, dy in directions:
        nx_pos, ny_pos = x + dx, y + dy
        if 0 <= nx_pos < height and 0 <= ny_pos < width:
            neighbor = grid[nx_pos][ny_pos]
            if not neighbor.blocked:
                neighbors.append(neighbor)
    return neighbors

# Function to find the nearest unblocked node to a given position
def find_nearest_unblocked_node(grid, x, y):
    height = len(grid)
    width = len(grid[0])
    visited = [[False]*width for _ in range(height)]
    queue = deque()
    queue.append((x, y))
    visited[x][y] = True
    while queue:
        cx, cy = queue.popleft()
        if not grid[cx][cy].blocked:
            return grid[cx][cy]
        # Explore neighbors
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx_pos, ny_pos = cx + dx, cy + dy
            if 0 <= nx_pos < height and 0 <= ny_pos < width and not visited[nx_pos][ny_pos]:
                visited[nx_pos][ny_pos] = True
                queue.append((nx_pos, ny_pos))
    # If no unblocked node is found
    return None

# Modified A* algorithm as a generator
def astar_generator(start: Node, goal: Node, grid):
    # Open list with start node
    open_list = []
    heapq.heappush(open_list, (0, start))

    # Visited nodes and their parents
    came_from = {}

    # Cost from start to node
    g_score = {start: 0}

    # For visualization
    closed_set = set()

    while open_list:
        # Node with the lowest f(n) = g(n) + h(n)
        current = heapq.heappop(open_list)[1]
        closed_set.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]

            # Yield final data
            yield {'grid': grid, 'start': start, 'goal': goal, 'path': path, 'closed_set': closed_set}

            # Pause and flash the path
            flash_colors = ['#ffd700', '#ff00ff']  # Gold and magenta
            for i in range(60):  # Pause for 3 seconds at 20 fps
                flash_color = flash_colors[i % len(flash_colors)]
                yield {'grid': grid, 'start': start, 'goal': goal, 'path': path, 'closed_set': closed_set, 'flash': True, 'flash_color': flash_color}

            return  # Path found

        yield {'grid': grid, 'start': start, 'goal': goal, 'path': None, 'closed_set': closed_set}

        neighbors = get_neighbors(current, grid)
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + neighbor.cost
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + h(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))
    # No path found
    for i in range(60):  # Pause for 3 seconds at 20 fps
        yield {'grid': grid, 'start': start, 'goal': goal, 'path': None, 'closed_set': closed_set, 'no_path': True}
    return

# Infinite generator to continuously generate new grids and A* searches
def infinite_astar_generator():
    while True:
        grid = generate_random_grid(width, height, obstacle_prob=0.2)
        start_node = grid[0][0]
        goal_node = grid[height - 1][width - 1]

        # Ensure start and goal are not blocked
        if start_node.blocked:
            start_node = find_nearest_unblocked_node(grid, 0, 0)
            if start_node is None:
                continue  # Skip to next iteration
        if goal_node.blocked:
            goal_node = find_nearest_unblocked_node(grid, height - 1, width - 1)
            if goal_node is None:
                continue  # Skip to next iteration

        # Create the A* generator
        astar_gen = astar_generator(start_node, goal_node, grid)

        # Yield data from the astar_gen
        for data in astar_gen:
            yield data

# Visualization function
def visualize_astar():
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.2)

    def init():
        ax.clear()
        ax.set_title('A* Pathfinding Visualization')
        ax.set_axis_off()

    # Update function for animation
    def update(data):
        ax.clear()
        ax.set_title('A* Pathfinding Visualization')
        ax.set_axis_off()
        grid = data['grid']
        start = data['start']
        goal = data['goal']
        path = data.get('path')
        closed_set = data.get('closed_set')

        # Build G and pos from grid
        G = nx.Graph()
        for row in grid:
            for node in row:
                G.add_node((node.x, node.y))

        for row in grid:
            for node in row:
                if node.blocked:
                    continue
                neighbors = get_neighbors(node, grid)
                for neighbor in neighbors:
                    G.add_edge((node.x, node.y), (neighbor.x, neighbor.y), weight=neighbor.cost)

        pos = {(node.x, node.y): (node.y, -node.x) for row in grid for node in row}

        draw_grid(ax, G, pos, data)

    def draw_grid(ax, G, pos, data):
        grid = data['grid']
        start = data['start']
        goal = data['goal']
        path = data.get('path')
        closed_set = data.get('closed_set')
        flash = data.get('flash', False)
        flash_color = data.get('flash_color', '#ffd700')  # Default to gold
        no_path = data.get('no_path', False)

        # Draw terrain nodes
        terrain_nodes = []
        terrain_colors = []
        for node in G.nodes():
            grid_node = grid[node[0]][node[1]]
            if not grid_node.blocked:
                if grid_node == start or grid_node == goal or (path and grid_node in path) or (closed_set and grid_node in closed_set):
                    continue
                cost = grid_node.cost
                terrain_nodes.append(node)
                if cost <= 1.0:
                    terrain_colors.append('#42f5e6')  # Normal terrain
                elif cost <= 2.0:
                    terrain_colors.append('#add8e6')  # Moderate terrain
                else:
                    terrain_colors.append('#ffa07a')  # Difficult terrain

        nx.draw_networkx_nodes(G, pos, nodelist=terrain_nodes, node_color=terrain_colors, node_size=100, ax=ax)

        # Draw obstacles
        obstacle_nodes = [node for node in G.nodes() if grid[node[0]][node[1]].blocked]
        nx.draw_networkx_nodes(G, pos, nodelist=obstacle_nodes, node_color='#2f4f4f', node_size=100, ax=ax)

        # Draw explored nodes
        if closed_set:
            explored_nodes = [ (node.x, node.y) for node in closed_set if node not in [start, goal] and (path is None or node not in path) ]
            nx.draw_networkx_nodes(G, pos, nodelist=explored_nodes, node_color='#f242f5', node_size=100, ax=ax)

        # Draw path nodes
        if path:
            path_nodes = [ (node.x, node.y) for node in path if node not in [start, goal] ]
            node_color = flash_color if flash else '#ffd700'
            nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color=node_color, node_size=100, ax=ax)
        else:
            if no_path:
                ax.text(0.5, 0.5, 'No Path Found', transform=ax.transAxes, ha='center', va='center', fontsize=24, color='red')

        # Draw start and goal nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[(start.x, start.y)], node_color='#00ff00', node_size=100, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[(goal.x, goal.y)], node_color='#ff0000', node_size=100, ax=ax)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax)

        # Highlight the path edges
        if path and len(path) > 1:
            edge_list = [((node.x, node.y), (path[i+1].x, path[i+1].y)) for i, node in enumerate(path[:-1])]
            edge_color = flash_color if flash else '#ffd700'
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_color, width=2, ax=ax)

        # Create a legend
        legend_elements = [
            mpatches.Patch(color='#00ff00', label='Start Node'),
            mpatches.Patch(color='#ff0000', label='Goal Node'),
            mpatches.Patch(color='#ffd700', label='Path'),
            mpatches.Patch(color='#2f4f4f', label='Blocked Nodes'),
            mpatches.Patch(color='#f242f5', label='Explored Nodes'),
            mpatches.Patch(color='#42f5e6', label='Normal Terrain (Cost 1.0)'),
            mpatches.Patch(color='#add8e6', label='Moderate Terrain (Cost 1.1 - 2.0)'),
            mpatches.Patch(color='#ffa07a', label='Difficult Terrain (Cost > 2.0)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    ani = animation.FuncAnimation(fig, update, frames=infinite_astar_generator(), init_func=init, repeat=False, blit=False, interval=50, cache_frame_data=False)

    plt.show()

# Function to generate a random grid
def generate_random_grid(width, height, obstacle_prob=0.2, terrain_costs=(1.0, 1.5, 2.0, 3.0), cost_probs=(0.6, 0.2, 0.15, 0.05)):
    grid = []
    for x in range(height):
        row = []
        for y in range(width):
            is_blocked = random.random() < obstacle_prob
            if is_blocked:
                node = Node(x, y, blocked=True)
            else:
                cost = random.choices(terrain_costs, weights=cost_probs, k=1)[0]
                node = Node(x, y, cost=cost)
            row.append(node)
        grid.append(row)
    return grid

if __name__ == "__main__":
    width = 30
    height = 30
    visualize_astar()
