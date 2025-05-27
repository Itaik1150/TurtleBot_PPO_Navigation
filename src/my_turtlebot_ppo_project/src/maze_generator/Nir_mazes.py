import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

# Parameters
map_size_m = 1.0
resolution = 0.05  # יותר גס = פחות קירות
grid_size = int(map_size_m / resolution)
wall_height = 0.2
cell_size = resolution

# Output directory
worlds_dir = Path.home() / "my_gazebo_worlds"
os.makedirs(worlds_dir, exist_ok=True)

def generate_path_with_turns(turns, grid_size):
    path = [(random.randint(1, grid_size - 2), random.randint(1, grid_size - 2))]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    last_dir = random.choice(directions)

    for _ in range(turns):
        next_dirs = [d for d in directions if d != last_dir and d != (-last_dir[0], -last_dir[1])]
        dir = random.choice(next_dirs)
        steps = random.randint(5, 10)
        for _ in range(steps):
            last = path[-1]
            next_pos = (last[0] + dir[0], last[1] + dir[1])
            if 1 <= next_pos[0] < grid_size - 1 and 1 <= next_pos[1] < grid_size - 1:
                path.append(next_pos)
            else:
                break
        last_dir = dir

    return path

def generate_maze_array(path_coords, grid_size):
    maze = np.ones((grid_size, grid_size), dtype=np.uint8)
    for x, y in path_coords:
        maze[y, x] = 0
    return maze

def create_world_file(maze_array, world_name):
    sdf = ET.Element("sdf", version="1.6")
    world = ET.SubElement(sdf, "world", name="default")

    # Add ground
    include = ET.SubElement(world, "include")
    uri = ET.SubElement(include, "uri")
    uri.text = "model://ground_plane"

    # Add light
    light = ET.SubElement(world, "light", name="sun", type="directional")
    ET.SubElement(light, "cast_shadows").text = "true"
    ET.SubElement(light, "pose").text = "0 0 10 0 0 0"
    direction = ET.SubElement(light, "direction")
    direction.text = "-0.5 0.5 -1"

    # Add horizontal walls
    for y in range(grid_size):
        start_x = None
        for x in range(grid_size):
            if maze_array[y, x] == 1:
                if start_x is None:
                    start_x = x
            else:
                if start_x is not None:
                    length = (x - start_x) * cell_size
                    mid_x = (start_x + x - 1) / 2 * cell_size
                    mid_y = y * cell_size
                    add_wall(world, mid_x, mid_y, length, cell_size)
                    start_x = None
        if start_x is not None:
            length = (grid_size - start_x) * cell_size
            mid_x = (start_x + grid_size - 1) / 2 * cell_size
            mid_y = y * cell_size
            add_wall(world, mid_x, mid_y, length, cell_size)

    # Add vertical walls
    for x in range(grid_size):
        start_y = None
        for y in range(grid_size):
            if maze_array[y, x] == 1:
                if start_y is None:
                    start_y = y
            else:
                if start_y is not None:
                    length = (y - start_y) * cell_size
                    mid_x = x * cell_size
                    mid_y = (start_y + y - 1) / 2 * cell_size
                    add_wall(world, mid_x, mid_y, cell_size, length)
                    start_y = None
        if start_y is not None:
            length = (grid_size - start_y) * cell_size
            mid_x = x * cell_size
            mid_y = (start_y + grid_size - 1) / 2 * cell_size
            add_wall(world, mid_x, mid_y, cell_size, length)

    file_path = worlds_dir / f"{world_name}.world"
    tree = ET.ElementTree(sdf)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)
    print(f"[✓] Created: {file_path}")

def add_wall(world, x, y, size_x, size_y):
    model = ET.SubElement(world, "model", name=f"wall_{x:.2f}_{y:.2f}")
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x} {y} {wall_height/2} 0 0 0"
    link = ET.SubElement(model, "link", name="link")
    collision = ET.SubElement(link, "collision", name="collision")
    geometry = ET.SubElement(collision, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = f"{size_x} {size_y} {wall_height}"
    visual = ET.SubElement(link, "visual", name="visual")
    geometry_v = ET.SubElement(visual, "geometry")
    box_v = ET.SubElement(geometry_v, "box")
    ET.SubElement(box_v, "size").text = f"{size_x} {size_y} {wall_height}"

# Create 20 worlds with increasing number of turns
for turns in range(2, 21):
    path = generate_path_with_turns(turns, grid_size)
    maze = generate_maze_array(path, grid_size)
    create_world_file(maze, f"maze_turns_{turns}")

# Zip worlds
zip_path = worlds_dir / "gazebo_mazes.zip"
with ZipFile(zip_path, 'w') as zipf:
    for file in worlds_dir.glob("maze_turns_*.world"):
        zipf.write(file, arcname=file.name)

print(f"[✓] Zipped all .world files to: {zip_path}")
