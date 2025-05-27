import os
import math

# הגדרות
model_name = "maze_two_turns"
model_path = os.path.expanduser(f"~/.gazebo/models/{model_name}")
os.makedirs(model_path, exist_ok=True)

wall_length = 53.3
wall_height = 1.0
wall_width = 0.2

# תוכן קובץ SDF של המודל (3 קירות בזווית)
sdf_template = """
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    {links}
  </model>
</sdf>
"""

link_template = """
    <link name="wall_{i}">
      <pose>{x} {y} {z} 0 0 {yaw}</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>{length} {width} {height}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{length} {width} {height}</size>
          </box>
        </geometry>
      </visual>
    </link>
"""

links = []

# קיר ראשון - לאורך ציר X
links.append(link_template.format(
    i=1,
    length=wall_length,
    width=wall_width,
    height=wall_height,
    x=wall_length / 2,
    y=0,
    z=wall_height / 2,
    yaw=0
))

# קיר שני - לאורך ציר Y
links.append(link_template.format(
    i=2,
    length=wall_length,
    width=wall_width,
    height=wall_height,
    x=wall_length,
    y=wall_length / 2,
    z=wall_height / 2,
    yaw=math.pi / 2
))

# קיר שלישי - שוב לאורך ציר X
links.append(link_template.format(
    i=3,
    length=wall_length,
    width=wall_width,
    height=wall_height,
    x=wall_length + wall_length / 2,
    y=wall_length,
    z=wall_height / 2,
    yaw=0
))

sdf_content = sdf_template.format(model_name=model_name, links="\n".join(links))

# כתיבת model.sdf
with open(os.path.join(model_path, "model.sdf"), "w") as f:
    f.write(sdf_content)

# כתיבת model.config
config_content = f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your@email.com</email>
  </author>
  <description>Maze model with 2 turns</description>
</model>
"""

with open(os.path.join(model_path, "model.config"), "w") as f:
    f.write(config_content)

print(f"[✓] Model '{model_name}' created at: {model_path}")
print("[i] You can now insert it via the 'Insert' tab in Gazebo GUI.")
