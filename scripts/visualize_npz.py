#!/usr/bin/env python3
"""
Visualize NPZ data as PNG
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import sys

def angle_difference_radians(angle1_rad, angle2_rad):
    # Calculate the absolute difference
    diff = abs(angle1_rad - angle2_rad)
    # Normalize to the range [0, 2*pi)
    normalized_diff = diff % (2 * math.pi)
    # Return the shortest angle
    if normalized_diff > math.pi:
        return (2 * math.pi) - normalized_diff
    else:
        return normalized_diff


def visualize_npz(npz_path, output_path=None):
    data = np.load(npz_path)
    
    token = str(data.get('token', 'scene'))
    if output_path is None:
        output_path = f"{token}_viz.png"
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Set range [-50, 50] for both axes
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'NPZ Visualization - {token}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # ========== 1. Lanes (boundaries only, no centerline) ==========
    # Skip route lanes completely as requested
    lanes = data['lanes']  # (70, 20, 12)
    route_lanes = data.get('route_lanes', None)
    
    # 不绘制 route lanes，只绘制普通车道边界
    for lane_idx in range(lanes.shape[0]):
        lane_x = lanes[lane_idx, :, 0]
        lane_y = lanes[lane_idx, :, 1]
        
        # Skip if all zeros
        if np.all(lane_x == 0) and np.all(lane_y == 0):
            continue
        
        # Skip if too far
        dist = np.sqrt(lane_x**2 + lane_y**2)
        if np.min(dist[dist > 0]) > 50:
            continue
        
        # Left boundary (blue)
        left_x = lanes[lane_idx, :, 0] + lanes[lane_idx, :, 4]
        left_y = lanes[lane_idx, :, 1] + lanes[lane_idx, :, 5]
        valid_left = (left_x != 0) | (left_y != 0)
        ax.plot(left_x[valid_left], left_y[valid_left], 'b--', linewidth=1, alpha=0.5)
        
        # Right boundary (red)
        right_x = lanes[lane_idx, :, 0] + lanes[lane_idx, :, 6]
        right_y = lanes[lane_idx, :, 1] + lanes[lane_idx, :, 7]
        valid_right = (right_x != 0) | (right_y != 0)
        ax.plot(right_x[valid_right], right_y[valid_right], 'r--', linewidth=1, alpha=0.5)
    
    # ========== 2. Ego (arrow at origin) ==========
    ego_state = data['ego_current_state']
    cos_h, sin_h = ego_state[2], ego_state[3]
    ego_heading = np.arctan2(sin_h, cos_h)
    # print("ego heading = ", heading)
    
    # Draw ego as an arrow showing direction
    arrow_length = 4
    # arrow_dx = cos_h * arrow_length
    # arrow_dy = sin_h * arrow_length
    
    # Draw arrow using FancyArrowPatch
    arrow = FancyArrowPatch(
        (0, 0), (arrow_length, 0),
        arrowstyle='-|>',
        mutation_scale=15,
        facecolor='red', edgecolor='darkred', linewidth=2
    )
    ax.add_patch(arrow)
    
    # Also add a small circle at ego position
    ego_circle = patches.Circle((0, 0), radius=1.5, facecolor='red', edgecolor='darkred', linewidth=2)
    ax.add_patch(ego_circle)
    


    ego_future = data['ego_agent_future']
    print(ego_future)
    if ego_future is not None:
        # Future trajectory: green thick line (3px)
        ax.plot(ego_future[:, 0], ego_future[:, 1], 'b-', linewidth=3, alpha=0.8)
        # # Start marker: circle (●) - same as past end
        # ax.plot(curr_x, curr_y, 'o', markersize=8, color='green', markeredgecolor='darkgreen', markeredgewidth=2)
        # # End marker: star (★)
        # ax.plot(end_x, end_y, '*', markersize=12, color='green', markeredgecolor='darkgreen', markeredgewidth=1)
    
    # ========== 2.1 Ego past trajectory ==========
    # ========== 2.1 Ego past trajectory ==========
    if 'ego_past' in data.files:
        ego_past = data['ego_past']
        # 绘制连续的历史轨迹
        valid_mask = (ego_past[:, 0] != 0) | (ego_past[:, 1] != 0)
        if np.any(valid_mask):
            ax.plot(ego_past[valid_mask, 0], ego_past[valid_mask, 1], 
                    'g--', linewidth=3, alpha=0.8, label='Ego Past')
    
    # Load neighbor agents data
    neighbor_past = data['neighbor_agents_past']
    neighbor_future = data.get('neighbor_agents_future')
    
    # Assign IDs to valid neighbor agents
    agent_id = 0
    agent_ids = {}  # agent_idx -> display_id
    
    # ========== 3. Neighbor agents past trajectories (blue thick + start/end markers) ==========
    for agent_idx in range(neighbor_past.shape[0]):  # Include all neighbors
        # Get current position (last timestep of past)
        curr_x = neighbor_past[agent_idx, -1, 0]
        curr_y = neighbor_past[agent_idx, -1, 1]
        
        # Get past trajectory
        past = neighbor_past[agent_idx, :, 0:2]
        
        # Check if valid (non-zero) - use threshold instead of exact comparison
        if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
            continue
        if abs(curr_x) > 50 or abs(curr_y) > 50:
            continue
        
        # Check if past has valid data
        if not np.any(past != 0):
            continue
        
        # Assign ID to this agent
        agent_ids[agent_idx] = agent_id
        agent_id += 1
        
        # Get first valid point as start
        valid_mask = (past[:, 0] != 0) | (past[:, 1] != 0)
        if not np.any(valid_mask):
            continue
        start_idx = np.argmax(valid_mask)
        start_x, start_y = past[start_idx, 0], past[start_idx, 1]

        # Draw agent ID near current position
        ax.annotate(str(agent_ids[agent_idx]), (curr_x, curr_y), 
                   fontsize=10, fontweight='bold', color='red',
                   ha='left', va='bottom', 
                   xytext=(5, 5), textcoords='offset points')
        
        # Past trajectory: blue thick line (3px) - only plot valid portion
        ax.plot(past[start_idx:, 0], past[start_idx:, 1], 'b-', linewidth=3, alpha=0.8)
        # Start marker: square (□)
        ax.plot(start_x, start_y, 's', markersize=8, color='blue', markeredgecolor='darkblue', markeredgewidth=2)
        # End marker: circle (●)
        ax.plot(curr_x, curr_y, 'o', markersize=8, color='blue', markeredgecolor='darkblue', markeredgewidth=2)
    
    # ========== 4. Neighbor agents future trajectories (green thick + start/end markers) ==========
    if neighbor_future is not None:
        for agent_idx in range(neighbor_future.shape[0]):  # Include all neighbors
            # Get current position as start
            curr_x = neighbor_past[agent_idx, -1, 0]
            curr_y = neighbor_past[agent_idx, -1, 1]
            
            # Get future trajectory
            future = neighbor_future[agent_idx, :, 0:2]
            
            # Check if valid
            if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
                continue
            if abs(curr_x) > 50 or abs(curr_y) > 50:
                continue
            
            # Check if future has valid data
            if not np.any(future != 0):
                continue
            
            # Get last valid point as end
            valid_mask = (future[:, 0] != 0) | (future[:, 1] != 0)
            if not np.any(valid_mask):
                continue
            end_idx = np.argmax(valid_mask)
            end_x, end_y = future[end_idx, 0], future[end_idx, 1]
            
            # Future trajectory: green thick line (3px) - only plot valid portion
            ax.plot(future[:, 0], future[:, 1], 'g-', linewidth=3, alpha=0.8)
            # Start marker: circle (●) - same as past end
            ax.plot(curr_x, curr_y, 'o', markersize=8, color='green', markeredgecolor='darkgreen', markeredgewidth=2)
            # End marker: star (★)
            ax.plot(end_x, end_y, '*', markersize=12, color='green', markeredgecolor='darkgreen', markeredgewidth=1)
    
    # ========== 5. Neighbor agents with box visualization ==========
    
    for agent_idx in range(neighbor_past.shape[0]):  # Include all neighbors
        # Check agent type
        type_v = neighbor_past[agent_idx, -1, 8]
        type_p = neighbor_past[agent_idx, -1, 9]
        type_b = neighbor_past[agent_idx, -1, 10]
        
        if type_v > 0.5:
            # Vehicle: blue rectangle
            agent_type = 'Vehicle'
            color = 'blue'
        elif type_p > 0.5:
            # Pedestrian: orange circle
            agent_type = 'Pedestrian'
            color = 'orange'
        elif type_b > 0.5:
            # Bicycle: purple diamond
            agent_type = 'Bicycle'
            color = 'purple'
        else:
            continue
        
        # Get current position (last timestep of past)
        curr_x = neighbor_past[agent_idx, -1, 0]
        curr_y = neighbor_past[agent_idx, -1, 1]
        
        # Skip if position is zero
        if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
            continue
        
        # Skip if too far
        if abs(curr_x) > 50 or abs(curr_y) > 50:
            continue
        
        # Get width (dim 6) and length (dim 7)
        width = neighbor_past[agent_idx, -1, 6]
        length = neighbor_past[agent_idx, -1, 7]
        
        # Get heading from cos_h (dim 2) and sin_h (dim 3)
        cos_h = neighbor_past[agent_idx, -1, 2]
        sin_h = neighbor_past[agent_idx, -1, 3]
        neighbor_heading = angle_difference_radians(ego_heading, np.arctan2(sin_h, cos_h))
        
        # Default size if zeros
        if width == 0:
            width = 2.0
        if length == 0:
            length = 4.0
        
        if agent_type == 'Vehicle':
            # Rectangle
            rect = patches.Rectangle(
                (curr_x - length/2, curr_y - width/2),
                length, width,
                angle= math.degrees(neighbor_heading),
                facecolor=color, edgecolor='black', linewidth=1, alpha=0.7
            )
            ax.add_patch(rect)
        elif agent_type == 'Pedestrian':
            # Circle
            radius = max(width, length) / 2
            circle = patches.Circle(
                (curr_x, curr_y), radius,
                facecolor=color, edgecolor='black', linewidth=1, alpha=0.7
            )
            ax.add_patch(circle)
        elif agent_type == 'Bicycle':
            # Diamond (rotated square)
            diamond = patches.RegularPolygon(
                (curr_x, curr_y), numVertices=4, radius=max(width, length)/2,
                orientation=neighbor_heading,
                facecolor=color, edgecolor='black', linewidth=1, alpha=0.7
            )
            ax.add_patch(diamond)
        # obs_length = 3.0
        # obs_arrow = FancyArrowPatch(
        #     (curr_x, curr_x), (sin(), 0),
        #     arrowstyle='-|>',
        #     mutation_scale=15,
        #     facecolor='green', edgecolor='darkred', linewidth=2
        # )
        # ax.add_patch(obs_arrow)
        
    # ========== 6. Static objects (gray triangles) ==========
    static_objs = data.get('static_objects')
    if static_objs is not None and len(static_objs) > 0:
        for obj_idx in range(static_objs.shape[0]):
            x, y = static_objs[obj_idx, 0], static_objs[obj_idx, 1]
            if x == 0 and y == 0:
                continue
            if abs(x) > 50 or abs(y) > 50:
                continue
            ax.plot(x, y, color='red', marker='^', markersize=20, alpha=0.7)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_npz.py <input.npz> [output.png]")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_npz(npz_path, output_path)
