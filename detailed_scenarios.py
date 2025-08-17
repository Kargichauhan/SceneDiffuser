import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
    print("✓ Saved detailed_scenario_analysis.png")

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
        ax.plot(x1, y1, 'b-', linewidth=3, alpha=0.7, label='Vehicle 1 path')
        ax.plot(x2, y2, 'purple', linestyle='-', linewidth=3, alpha=0.7, label='Vehicle 2 path')
        
        # Show vehicles at collision point
        ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='Collision')
        ax.add_patch(plt.Rectangle((-1.5, -0.8), 3, 1.6, color='blue', alpha=0.7))
        ax.add_patch(plt.Rectangle((-0.8, -1.5), 1.6, 3, color='purple', alpha=0.7))
        
        # Add collision indicator
        ax.text(5, 8, 'COLLISION!', fontsize=14, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    else:  # ours
        # Show safe navigation
        # Vehicle 1 (horizontal) - slightly delayed
        vehicle1_trajectory = [(-15, 0), (-10, 0), (-5, 0), (3, 0)]
        # Vehicle 2 (vertical) - passes first
        vehicle2_trajectory = [(0, -15), (0, -10), (0, -5), (0, 3)]
        
        x1, y1 = zip(*vehicle1_trajectory)
        x2, y2 = zip(*vehicle2_trajectory)
        ax.plot(x1, y1, 'g-', linewidth=3, alpha=0.7, label='Vehicle 1 path')
        ax.plot(x2, y2, 'darkgreen', linestyle='-', linewidth=3, alpha=0.7, label='Vehicle 2 path')
        
        # Show vehicles safely spaced
        ax.add_patch(plt.Rectangle((2, -0.8), 3, 1.6, color='green', alpha=0.7))
        ax.add_patch(plt.Rectangle((-0.8, 2), 1.6, 3, color='darkgreen', alpha=0.7))
        
        # Add arrows showing energy forces
        ax.arrow(-3, 0, -2, 0, head_width=1, head_length=1, fc='orange', ec='orange', linewidth=2)
        ax.arrow(0, -3, 0, -2, head_width=1, head_length=1, fc='orange', ec='orange', linewidth=2)
        ax.text(-12, 8, 'Repulsive Forces', fontsize=12, color='orange', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.text(8, 8, 'SAFE PASSAGE', fontsize=14, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

def create_merge_scenario(ax, method):
    """Create highway merge scenario"""
    # Draw main highway
    ax.add_patch(plt.Rectangle((-30, -2), 60, 4, color='lightblue', alpha=0.3))
    # Draw merge lane
    ax.add_patch(plt.Rectangle((10, 2), 20, 3, color='lightblue', alpha=0.3))
    
    # Add lane markings
    ax.plot([-30, 30], [0, 0], 'w--', linewidth=1, alpha=0.7)
    ax.plot([10, 30], [2, 2], 'w--', linewidth=1, alpha=0.7)
    ax.plot([10, 30], [5, 5], 'w-', linewidth=2, alpha=0.7)  # Top boundary
    ax.plot([-30, 30], [2, 2], 'w-', linewidth=2, alpha=0.7)  # Bottom boundary
    ax.plot([-30, 30], [-2, -2], 'w-', linewidth=2, alpha=0.7)
    
    if method == 'baseline':
        # Show unsafe merge
        # Main highway vehicles
        highway_vehicles = [(-20, 0), (-10, 0), (5, 0), (15, 0)]
        # Merging vehicle - collision course
        merge_vehicle = [(12, 3.5), (14, 2.5), (16, 1.5), (17, 0)]
        
        # Draw highway vehicles
        for i, (x, y) in enumerate(highway_vehicles):
            ax.add_patch(plt.Rectangle((x-1.5, y-0.8), 3, 1.6, color='blue', alpha=0.7))
            ax.text(x, y+2, f'V{i+1}', ha='center', fontsize=8, fontweight='bold')
        
        # Draw merge trajectory
        mx, my = zip(*merge_vehicle)
        ax.plot(mx, my, 'purple', linestyle='-', linewidth=3, alpha=0.8)
        ax.add_patch(plt.Rectangle((16, -0.8), 3, 1.6, color='purple', alpha=0.7))
        
        # Show collision
        ax.plot(17, 0, 'rx', markersize=15, markeredgewidth=3)
        ax.text(20, 4, 'UNSAFE MERGE', fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add danger zone
        ax.add_patch(plt.Circle((16.5, 0), 3, fill=False, color='red', linestyle='--', linewidth=2))
        ax.text(12, -4, 'Danger Zone', fontsize=10, color='red', fontweight='bold')
        
    else:  # ours
        # Show safe merge with gap finding
        highway_vehicles = [(-20, 0), (-10, 0), (6, 0), (22, 0)]  # Larger gap
        merge_vehicle = [(12, 3.5), (13, 2.8), (14, 2.0), (15, 1.2), (16, 0.4), (17, 0)]
        
        # Draw highway vehicles
        for i, (x, y) in enumerate(highway_vehicles):
            ax.add_patch(plt.Rectangle((x-1.5, y-0.8), 3, 1.6, color='green', alpha=0.7))
            ax.text(x, y+2.5, f'V{i+1}', ha='center', fontsize=8, fontweight='bold')
        
        # Draw smooth merge
        mx, my = zip(*merge_vehicle)
        ax.plot(mx, my, 'darkgreen', linestyle='-', linewidth=3, alpha=0.8)
        ax.add_patch(plt.Rectangle((16, -0.8), 3, 1.6, color='darkgreen', alpha=0.7))
        
        # Show energy guidance and gap detection
        ax.arrow(14, 0, 4, 0, head_width=0.5, head_length=1, fc='orange', ec='orange', linewidth=2)
        ax.text(10, 4.5, 'Gap Detection', fontsize=11, color='orange', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(20, -4, 'SAFE MERGE', fontsize=12, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add safe zone
        ax.add_patch(plt.Circle((16.5, 0), 4, fill=False, color='green', linestyle='--', linewidth=2))
        ax.text(12, -3, 'Safe Zone', fontsize=10, color='green', fontweight='bold')
    
    ax.set_xlim(-25, 35)
    ax.set_ylim(-6, 7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def create_comprehensive_appendix_figure():
    """Create a comprehensive 6-panel appendix figure"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Row 1: Collision scenarios
    create_collision_scenario(axes[0, 0], 'baseline')
    axes[0, 0].set_title('(a) Baseline: Intersection Collision', fontsize=12, fontweight='bold')
    
    create_collision_scenario(axes[0, 1], 'ours')
    axes[0, 1].set_title('(b) Ours: Safe Intersection Navigation', fontsize=12, fontweight='bold')
    
    # Row 2: Merge scenarios  
    create_merge_scenario(axes[1, 0], 'baseline')
    axes[1, 0].set_title('(c) Baseline: Unsafe Highway Merge', fontsize=12, fontweight='bold')
    
    create_merge_scenario(axes[1, 1], 'ours')
    axes[1, 1].set_title('(d) Ours: Constraint-Guided Merge', fontsize=12, fontweight='bold')
    
    # Row 3: Additional scenarios
    create_roundabout_scenario(axes[2, 0], 'baseline')
    axes[2, 0].set_title('(e) Baseline: Roundabout Violations', fontsize=12, fontweight='bold')
    
    create_roundabout_scenario(axes[2, 1], 'ours')
    axes[2, 1].set_title('(f) Ours: Smooth Roundabout Flow', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_appendix_scenarios.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comprehensive_appendix_scenarios.png")

def create_roundabout_scenario(ax, method):
    """Create roundabout scenario"""
    # Draw roundabout
    outer_circle = plt.Circle((0, 0), 10, fill=False, color='gray', linewidth=3)
    inner_circle = plt.Circle((0, 0), 6, fill=True, color='lightgray', alpha=0.5)
    ax.add_patch(outer_circle)
    ax.add_patch(inner_circle)
    
    # Add entry/exit points
    entry_points = [(0, 12), (12, 0), (0, -12), (-12, 0)]
    for x, y in entry_points:
        ax.add_patch(plt.Rectangle((x-1, y-8 if y > 0 else y), 2, 8, color='lightblue', alpha=0.3))
    
    if method == 'baseline':
        # Show chaotic roundabout behavior
        # Multiple vehicles with violations
        angles = np.linspace(0, 2*np.pi, 8)
        for i, angle in enumerate(angles):
            r = 8
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            
            if i % 3 == 0:  # Some collisions
                ax.plot(x, y, 'rx', markersize=12, markeredgewidth=3)
                ax.plot(x+0.5, y+0.5, 'rx', markersize=12, markeredgewidth=3)
            else:
                ax.add_patch(plt.Rectangle((x-1, y-0.5), 2, 1, color='blue', alpha=0.7))
        
        ax.text(0, 15, 'CHAOTIC FLOW', fontsize=12, color='red', fontweight='bold', ha='center')
        
    else:  # ours
        # Show organized flow
        angles = np.linspace(0, 2*np.pi, 6)
        for i, angle in enumerate(angles):
            r = 8
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            ax.add_patch(plt.Rectangle((x-1, y-0.5), 2, 1, color='green', alpha=0.7))
            
            # Add flow arrows
            ax.arrow(x, y, -2*np.sin(angle), 2*np.cos(angle), 
                    head_width=0.5, head_length=0.5, fc='orange', ec='orange')
        
        ax.text(0, 15, 'ORGANIZED FLOW', fontsize=12, color='green', fontweight='bold', ha='center')
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    print("Creating detailed scenario analysis...")
    create_detailed_scenario_analysis()
    
    print("Creating comprehensive appendix figure...")
    create_comprehensive_appendix_figure()
    
    print("Done! Generated files:")
    print("  - detailed_scenario_analysis.png")
    print("  - comprehensive_appendix_scenarios.png")
