# SinglePose Class

The `SinglePose` class represents a complete skeletal pose at a specific frame, containing transformation data for all joints in both world and parent-relative coordinate spaces.

## Class Definition

```python
@define(kw_only=True, eq=False)
class SinglePose:
    """Represents a single pose of a skeleton at a specific frame."""
```

!!! info "Implementation Notes"
    - Uses the `attrs` library with `@define` decorator
    - Keyword-only initialization (`kw_only=True`)
    - Equality comparison disabled (`eq=False`)

## Attributes

### Core Properties

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `joint_transforms_wrt_world` | `Dict[str, np.ndarray]` | `{}` | Joint transforms in world space |
| `joint_transforms_wrt_parent` | `Dict[str, np.ndarray]` | `{}` | Joint transforms relative to parent |
| `source_frame_index` | `int \| None` | `None` | Original frame index in motion sequence |

### Detailed Attribute Documentation

#### `joint_transforms_wrt_world: Dict[str, np.ndarray]`
A dictionary mapping joint names to their 4x4 transformation matrices in world space. Each joint's position and orientation are relative to the global coordinate system.

```python
# Example world transforms
world_transforms = {
    "Hips": np.array([[1, 0, 0, 0], [0, 1, 0, 1.2], [0, 0, 1, 0], [0, 0, 0, 1]]),
    "Spine": np.array([[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, 0], [0, 0, 0, 1]]),
    "LeftShoulder": np.array([[1, 0, 0, 0.3], [0, 1, 0, 1.7], [0, 0, 1, 0], [0, 0, 0, 1]])
}

pose = SinglePose(joint_transforms_wrt_world=world_transforms)
```

#### `joint_transforms_wrt_parent: Dict[str, np.ndarray]`
A dictionary mapping joint names to their 4x4 transformation matrices relative to their parent joint. This maintains the original hierarchical structure from the BVH file.

```python
# Example parent-relative transforms
parent_transforms = {
    "Hips": np.array([[1, 0, 0, 0], [0, 1, 0, 1.2], [0, 0, 1, 0], [0, 0, 0, 1]]),  # Root
    "Spine": np.array([[1, 0, 0, 0], [0, 1, 0, 0.3], [0, 0, 1, 0], [0, 0, 0, 1]]),  # 0.3 up from Hips
    "LeftShoulder": np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])  # Offset from Spine
}

pose = SinglePose(joint_transforms_wrt_parent=parent_transforms)
```

#### `source_frame_index: int | None`
The index of the frame this pose corresponds to in the original motion sequence. Useful for tracking and debugging.

```python
pose = SinglePose(
    joint_transforms_wrt_world=world_transforms,
    source_frame_index=42  # This pose is from frame 42
)
```

## Methods

### `get_joint_positions(relative_to: str = "world") -> Dict[str, np.ndarray]`

Extract 3D positions from transformation matrices.

#### Parameters
- `relative_to` (`str`, optional): Coordinate space - `"world"` or `"parent"`. Default: `"world"`

#### Returns
- `Dict[str, np.ndarray]`: Dictionary mapping joint names to 3D position vectors

#### Raises
- `ValueError`: If `relative_to` is not `"world"` or `"parent"`

#### Example Usage

```python
pose = SinglePose(joint_transforms_wrt_world=world_transforms)

# Get world positions
world_positions = pose.get_joint_positions("world")
print(world_positions["Hips"])  # [0.0, 1.2, 0.0]

# Get parent-relative positions
parent_positions = pose.get_joint_positions("parent")
print(parent_positions["Spine"])  # [0.0, 0.3, 0.0] (relative to Hips)
```

### `apply_transform(transform: np.ndarray, root_name: str) -> None`

Apply a transformation to the entire pose destructively.

#### Parameters
- `transform` (`np.ndarray`): 4x4 transformation matrix to apply
- `root_name` (`str`): Name of the root joint

#### Description
This method modifies the pose in-place by applying the given transformation to all joints. It updates both world transforms and the root's parent-relative transform.

#### Example Usage

```python
# Create a transformation (e.g., 90-degree rotation around Y-axis)
rotation_y = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1]
])

# Apply transformation to the entire pose
pose.apply_transform(rotation_y, "Hips")
```

## Usage Examples

### Creating a Complete Pose

```python
from hmodel_gen.bvh_parser import SinglePose
import numpy as np

# Define joint hierarchy and transforms
joints_data = {
    "Hips": {
        "world": np.eye(4),
        "parent": np.eye(4)  # Root joint
    },
    "Spine": {
        "world": np.array([[1, 0, 0, 0], [0, 1, 0, 0.3], [0, 0, 1, 0], [0, 0, 0, 1]]),
        "parent": np.array([[1, 0, 0, 0], [0, 1, 0, 0.3], [0, 0, 1, 0], [0, 0, 0, 1]])
    },
    "LeftShoulder": {
        "world": np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]]),
        "parent": np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
    }
}

# Create the pose
world_transforms = {name: data["world"] for name, data in joints_data.items()}
parent_transforms = {name: data["parent"] for name, data in joints_data.items()}

pose = SinglePose(
    joint_transforms_wrt_world=world_transforms,
    joint_transforms_wrt_parent=parent_transforms,
    source_frame_index=0
)
```

### Analyzing Joint Positions

```python
# Get all joint positions in world space
world_positions = pose.get_joint_positions("world")

# Calculate distances between joints
def calculate_bone_length(pose: SinglePose, parent_joint: str, child_joint: str) -> float:
    """Calculate the distance between two joints."""
    positions = pose.get_joint_positions("world")
    parent_pos = positions[parent_joint]
    child_pos = positions[child_joint]
    return np.linalg.norm(child_pos - parent_pos)

# Calculate spine length
spine_length = calculate_bone_length(pose, "Hips", "Spine")
print(f"Spine length: {spine_length:.3f}")

# Get center of mass (simplified)
def get_pose_center(pose: SinglePose) -> np.ndarray:
    """Calculate the center of all joint positions."""
    positions = pose.get_joint_positions("world")
    all_positions = np.array(list(positions.values()))
    return np.mean(all_positions, axis=0)

center = get_pose_center(pose)
print(f"Pose center: {center}")
```

### Pose Transformations

```python
# Translate entire pose
translation = np.eye(4)
translation[:3, 3] = [1.0, 0.5, 2.0]  # Move 1 unit in X, 0.5 in Y, 2 in Z

pose_copy = copy.deepcopy(pose)
pose_copy.apply_transform(translation, "Hips")

# Rotate entire pose around Y-axis
angle = np.pi / 4  # 45 degrees
rotation_y = np.array([
    [np.cos(angle), 0, np.sin(angle), 0],
    [0, 1, 0, 0],
    [-np.sin(angle), 0, np.cos(angle), 0],
    [0, 0, 0, 1]
])

pose_rotated = copy.deepcopy(pose)
pose_rotated.apply_transform(rotation_y, "Hips")

# Scale entire pose
scale_factor = 1.5
scale_matrix = np.eye(4) * scale_factor
scale_matrix[3, 3] = 1.0  # Keep homogeneous coordinate as 1

pose_scaled = copy.deepcopy(pose)
pose_scaled.apply_transform(scale_matrix, "Hips")
```

### Pose Comparison

```python
def compare_poses(pose1: SinglePose, pose2: SinglePose) -> Dict[str, float]:
    """Compare two poses by calculating joint position differences."""
    pos1 = pose1.get_joint_positions("world")
    pos2 = pose2.get_joint_positions("world")
    
    differences = {}
    common_joints = set(pos1.keys()) & set(pos2.keys())
    
    for joint_name in common_joints:
        diff = np.linalg.norm(pos1[joint_name] - pos2[joint_name])
        differences[joint_name] = diff
    
    return differences

# Compare original and transformed poses
differences = compare_poses(pose, pose_rotated)
print("Joint position differences after rotation:")
for joint, diff in differences.items():
    print(f"{joint}: {diff:.3f}")
```

## Advanced Usage Patterns

### Coordinate Space Validation

```python
def validate_pose_consistency(pose: SinglePose, skeleton_graph) -> bool:
    """Validate that world transforms are consistent with parent-relative transforms."""
    import networkx as nx
    
    # Get topological order of joints
    joint_order = list(nx.topological_sort(skeleton_graph))
    
    for joint_name in joint_order:
        if skeleton_graph.in_degree(joint_name) == 0:  # Root joint
            continue
            
        # Get parent
        parent_name = list(skeleton_graph.predecessors(joint_name))[0]
        
        # Check if world transform matches computation from parent
        expected_world = (pose.joint_transforms_wrt_world[parent_name] @ 
                         pose.joint_transforms_wrt_parent[joint_name])
        actual_world = pose.joint_transforms_wrt_world[joint_name]
        
        if not np.allclose(expected_world, actual_world, atol=1e-6):
            print(f"Inconsistency detected at joint: {joint_name}")
            return False
    
    return True
```

### Pose Interpolation

```python
def interpolate_poses(pose1: SinglePose, pose2: SinglePose, t: float) -> SinglePose:
    """Linear interpolation between two poses."""
    if not (0.0 <= t <= 1.0):
        raise ValueError("Interpolation parameter t must be between 0 and 1")
    
    # Get common joints
    common_joints = set(pose1.joint_transforms_wrt_world.keys()) & \
                   set(pose2.joint_transforms_wrt_world.keys())
    
    interpolated_world = {}
    interpolated_parent = {}
    
    for joint_name in common_joints:
        # Simple linear interpolation of transformation matrices
        # Note: This is not the best approach for rotations (should use quaternions)
        world1 = pose1.joint_transforms_wrt_world[joint_name]
        world2 = pose2.joint_transforms_wrt_world[joint_name]
        interpolated_world[joint_name] = (1 - t) * world1 + t * world2
        
        parent1 = pose1.joint_transforms_wrt_parent[joint_name]
        parent2 = pose2.joint_transforms_wrt_parent[joint_name]
        interpolated_parent[joint_name] = (1 - t) * parent1 + t * parent2
    
    return SinglePose(
        joint_transforms_wrt_world=interpolated_world,
        joint_transforms_wrt_parent=interpolated_parent,
        source_frame_index=None  # Interpolated frame has no source index
    )

# Usage
middle_pose = interpolate_poses(pose1, pose2, 0.5)  # 50% between poses
```

## Integration with BVH Classes

### From BVHData

```python
# BVHData provides SinglePose objects through get_pose()
bvh_data = BVHData.from_file("motion.bvh")
pose = bvh_data.get_pose(100)  # Returns a SinglePose object

# Access pose data
positions = pose.get_joint_positions("world")
frame_index = pose.source_frame_index
```

### With Visualization

```python
def visualize_pose_comparison(pose1: SinglePose, pose2: SinglePose):
    """Visualize two poses side by side."""
    import igpy.myplot.vistaplot as vplot
    
    plotter = vplot.ExPlotter.init_with_background_plotter()
    
    # Visualize first pose in blue
    pos1 = pose1.get_joint_positions("world")
    points1 = np.array(list(pos1.values()))
    plotter.add_point_cloud(points1, color3f=(0, 0, 1))
    
    # Visualize second pose in red
    pos2 = pose2.get_joint_positions("world")
    points2 = np.array(list(pos2.values()))
    plotter.add_point_cloud(points2, color3f=(1, 0, 0))
    
    return plotter
```

## Best Practices

!!! tip "Memory Management"
    - Use `copy.deepcopy()` when creating modified versions of poses
    - Consider the memory footprint when storing many poses
    - For large datasets, prefer `BVHFlatData` over collections of `SinglePose` objects

!!! warning "Transformation Consistency"
    - Always ensure world and parent-relative transforms are consistent
    - Use `apply_transform()` to maintain consistency when modifying poses
    - Validate pose data when loading from external sources

!!! info "Coordinate Space Usage"
    - Use world space for absolute positioning and visualization
    - Use parent-relative space for animation retargeting and modification
    - Be explicit about which coordinate space you're working in

## Common Pitfalls

### Inconsistent Transform Dictionaries
```python
# ❌ Problematic - mismatched joint sets
world_transforms = {"Hips": np.eye(4), "Spine": np.eye(4)}
parent_transforms = {"Hips": np.eye(4)}  # Missing "Spine"

pose = SinglePose(
    joint_transforms_wrt_world=world_transforms,
    joint_transforms_wrt_parent=parent_transforms  # Inconsistent!
)

# ✅ Correct - consistent joint sets
world_transforms = {"Hips": np.eye(4), "Spine": np.eye(4)}
parent_transforms = {"Hips": np.eye(4), "Spine": np.eye(4)}
```

### Incorrect Transform Application
```python
# ❌ Wrong - modifying individual transforms breaks consistency
pose.joint_transforms_wrt_world["Hips"] = new_transform  # Breaks consistency

# ✅ Correct - use apply_transform for global changes
pose.apply_transform(global_transform, "Hips")  # Maintains consistency
```

### Invalid Coordinate Space Parameters
```python
# ❌ Invalid coordinate space
positions = pose.get_joint_positions("global")  # Raises ValueError

# ✅ Valid coordinate spaces
world_pos = pose.get_joint_positions("world")
parent_pos = pose.get_joint_positions("parent")
```