def create_detailed_scenario_analysis():
    """Create detailed scenario analysis like the reference figure"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # (a) Collision Avoidance at Intersection
    ax = axes[0, 0]
    create_collision_scenario(ax, method='baseline')
    ax.set_title('(a) Baseline: Collision at intersection', fontsize=12, fontweight='bold')
    
    ax = axes[0, 1] 
    create_collision_scenario(ax, method='ours')
    ax.set_title('(a) Ours: Safe navigation through intersection', fontsize=12, fontweight='bold')
    
    # (b) Highway Merge
    ax = axes[1, 0]
    create_merge_scenario(ax, method='baseline')
    ax.set_title('(b) Baseline: Unsafe merge with overlaps', fontsize=12, fontweight='bold')
    
    ax = axes[1, 1]
    create_merge_scenario(ax, method='ours') 
    ax.set_title('(b) Ours: Smooth merge with proper spacing', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_scenario_analysis.png', dpi=300, bbox_inches='tight')

def create_collision_scenario(ax, method):
    """Create intersection collision scenario"""
    # Draw intersection
    ax.add_patch(plt.Rectangle((-20, -3), 40, 6, color='lightblue', alpha=0.3, label='Road'))
    ax.add_patch(plt.Rectangle((-3, -20), 6, 40, color='lightblue', alpha=0.3))
    
    # Add lane markings
    ax.plot([-20, 20], [0, 0], 'w--', linewidth=1, alpha=0.7)
    ax.plot([0, 0], [-20, 20], 'w--', linewidth=1, alpha=0.7)
    
    if method == 'baseline':
        # Show collision scenario
        # Vehicle 1 (horizontal)
        vehicle1_trajectory = [(-15, 0), (-10, 0), (-5, 0), (0, 0)]
        # Vehicle 2 (vertical) - collision course
        vehicle2_trajectory = [(0, -15), (0, -10), (0, -5), (0, 0)]
        
        # Draw trajectories
        x1, y1 = zip(*vehicle1_trajectory)
        x2, y2 = zip(*vehicle2_trajectory)
        ax.plot(x1, y1, 'b-', linewidth=2, alpha=0.7, label='Vehicle 1 path')
        ax.plot(x2, y2, 'purple', linestyle='-', linewidth=2, alpha=0.7, label='Vehicle 2 path')
        
        # Show vehicles at collision point
        ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='Collision')
        ax.add_patch(plt.Rectangle((-1, -0.5), 2, 1, color='blue', alpha=0.7))
        ax.add_patch(plt.Rectangle((-0.5, -1), 1, 2, color='purple', alpha=0.7))
        
        # Add collision indicator
        ax.text(2, 2, 'COLLISION!', fontsize=12, color='red', fontweight='bold')
        
    else:  # ours
        # Show safe navigation
        # Vehicle 1 (horizontal) - slightly delayed
        vehicle1_trajectory = [(-15, 0), (-10, 0), (-5, 0), (2, 0)]
        # Vehicle 2 (vertical) - passes first
        vehicle2_trajectory = [(0, -15), (0, -10), (0, -5), (0, 2)]
        
        x1, y1 = zip(*vehicle1_trajectory)
        x2, y2 = zip(*vehicle2_trajectory)
        ax.plot(x1, y1, 'g-', linewidth=2, alpha=0.7, label='Vehicle 1 path')
        ax.plot(x2, y2, 'darkgreen', linestyle='-', linewidth=2, alpha=0.7, label='Vehicle 2 path')
        
        # Show vehicles safely spaced
        ax.add_patch(plt.Rectangle((1, -0.5), 2, 1, color='green', alpha=0.7))
        ax.add_patch(plt.Rectangle((-0.5, 1), 1, 2, color='darkgreen', alpha=0.7))
        
        # Add arrows showing energy forces
        ax.arrow(-2, 0, -1, 0, head_width=0.5, head_length=0.5, fc='orange', ec='orange')
        ax.arrow(0, -2, 0, -1, head_width=0.5, head_length=0.5, fc='orange', ec='orange')
        ax.text(-8, 3, 'Repulsive Force', fontsize=10, color='orange', fontweight='bold')
        
        ax.text(5, 5, 'SAFE PASSAGE', fontsize=12, color='green', fontweight='bold')
    
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

def create_merge_scenario(ax, method):
    """Create highway merge scenario"""
    # Draw main highway
    ax.add_patch(plt.Rectangle((-30, -2), 60, 4, color='lightblue', alpha=0.3))
    # Draw merge lane
    ax.add_patch(plt.Rectangle((10, 2), 20, 3, color='lightblue', alpha=0.3))
    
    # Add lane markings
    ax.plot([-30, 30], [0, 0], 'w--', linewidth=1, alpha=0.7)
    ax.plot([10, 30], [2, 2], 'w--', linewidth=1, alpha=0.7)
    
    if method == 'baseline':
        # Show unsafe merge
        # Main highway vehicles
        highway_vehicles = [(-20, 0), (-10, 0), (5, 0), (15, 0)]
        # Merging vehicle - collision course
        merge_vehicle = [(12, 3), (14, 2.5), (16, 1.5), (16, 0)]
        
        # Draw vehicles
        for i, (x, y) in enumerate(highway_vehicles):
            ax.add_patch(plt.Rectangle((x-1, y-0.5), 2, 1, color='blue', alpha=0.7))
        
        # Draw merge trajectory
        mx, my = zip(*merge_vehicle)
        ax.plot(mx, my, 'purple', linestyle='-', linewidth=2, alpha=0.7)
        ax.add_patch(plt.Rectangle((15, -0.5), 2, 1, color='purple', alpha=0.7))
        
        # Show collision
        ax.plot(16, 0, 'rx', markersize=12, markeredgewidth=3)
        ax.text(18, 3, 'UNSAFE MERGE', fontsize=10, color='red', fontweight='bold')
        
    else:  # ours
        # Show safe merge with gap finding
        highway_vehicles = [(-20, 0), (-10, 0), (8, 0), (20, 0)]  # Larger gap
        merge_vehicle = [(12, 3), (13, 2.5), (14, 1.5), (15, 0.5), (16, 0)]
        
        # Draw vehicles
        for i, (x, y) in enumerate(highway_vehicles):
            ax.add_patch(plt.Rectangle((x-1, y-0.5), 2, 1, color='green', alpha=0.7))
        
        # Draw smooth merge
        mx, my = zip(*merge_vehicle)
        ax.plot(mx, my, 'darkgreen', linestyle='-', linewidth=2, alpha=0.7)
        ax.add_patch(plt.Rectangle((15, -0.5), 2, 1, color='darkgreen', alpha=0.7))
        
        # Show energy guidance
        ax.arrow(12, 0, 2, 0, head_width=0.3, head_length=0.5, fc='orange', ec='orange')
        ax.text(10, 4, 'Gap Detection', fontsize=10, color='orange', fontweight='bold')
        ax.text(18, -3, 'SAFE MERGE', fontsize=10, color='green', fontweight='bold')
    
    ax.set_xlim(-25, 35)
    ax.set_ylim(-5, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)