# SingleJoint Class

The `SingleJoint` class represents an individual joint in a skeletal hierarchy, storing both local and global transformation information for a specific joint at a particular frame.

## Class Definition

```python
@define(kw_only=True, eq=False)
class SingleJoint:
    """Represents a single joint in a skeleton."""
```

!!! info "Implementation Notes"
    - Uses the `attrs` library with `@define` decorator
    - Keyword-only initialization (`kw_only=True`)
    - Equality comparison disabled (`eq=False`)

## Attributes

### Core Properties

| Attribute | Type | Description |
|-----------|------|-------------|
| `joint_name` | `str` | The name of the joint |
| `parent_joint_name` | `str \| None` | The name of the parent joint (None for root) |
| `joint_transform_wrt_parent` | `np.ndarray` | 4x4 transformation matrix relative to parent |
| `joint_transform_wrt_world` | `np.ndarray` | 4x4 transformation matrix in world space |

### Detailed Attribute Documentation

#### `joint_name: str`
The unique identifier for this joint within the skeleton hierarchy.

```python
joint = SingleJoint(joint_name="LeftShoulder")
print(joint.joint_name)  # "LeftShoulder"
```

#### `parent_joint_name: str | None`
The name of the parent joint in the hierarchy. Set to `None` for the root joint.

```python
# Root joint
root = SingleJoint(joint_name="Hips", parent_joint_name=None)

# Child joint
shoulder = SingleJoint(
    joint_name="LeftShoulder", 
    parent_joint_name="Spine2"
)
```

#### `joint_transform_wrt_parent: np.ndarray`
A 4x4 homogeneous transformation matrix representing the joint's position and orientation relative to its parent joint.

**Default**: `np.eye(4)` (identity matrix)

```python
# Example transformation matrix
transform_local = np.array([
    [1.0, 0.0, 0.0, 0.1],  # X translation: 0.1
    [0.0, 1.0, 0.0, 0.2],  # Y translation: 0.2
    [0.0, 0.0, 1.0, 0.0],  # Z translation: 0.0
    [0.0, 0.0, 0.0, 1.0]   # Homogeneous coordinate
])

joint = SingleJoint(
    joint_name="LeftElbow",
    parent_joint_name="LeftShoulder",
    joint_transform_wrt_parent=transform_local
)
```

#### `joint_transform_wrt_world: np.ndarray`
A 4x4 homogeneous transformation matrix representing the joint's position and orientation in world space.

**Default**: `np.eye(4)` (identity matrix)

```python
# World transform is typically computed from parent transforms
joint = SingleJoint(
    joint_name="LeftElbow",
    joint_transform_wrt_world=world_transform_matrix
)
```

## Matrix Structure

The transformation matrices follow the standard 4x4 homogeneous transformation format:

```python
transform = [
    [r11, r12, r13, tx],  # First row: rotation + x translation
    [r21, r22, r23, ty],  # Second row: rotation + y translation  
    [r31, r32, r33, tz],  # Third row: rotation + z translation
    [0,   0,   0,   1 ]   # Fourth row: homogeneous coordinates
]
```

Where:
- `r11-r33`: 3x3 rotation matrix components
- `tx, ty, tz`: Translation vector components

## Usage Examples

### Basic Joint Creation

```python
from hmodel_gen.bvh_parser import SingleJoint
import numpy as np

# Create a root joint
root_joint = SingleJoint(
    joint_name="Hips",
    parent_joint_name=None,
    joint_transform_wrt_world=np.eye(4)
)

# Create a child joint with local offset
child_transform = np.eye(4)
child_transform[:3, 3] = [0.0, 0.5, 0.0]  # 0.5 units up from parent

child_joint = SingleJoint(
    joint_name="Spine",
    parent_joint_name="Hips",
    joint_transform_wrt_parent=child_transform
)
```

### Extracting Position and Rotation

```python
# Extract position from transformation matrix
def get_joint_position(joint: SingleJoint, space: str = "world") -> np.ndarray:
    """Extract 3D position from joint transformation."""
    if space == "world":
        return joint.joint_transform_wrt_world[:3, 3]
    else:
        return joint.joint_transform_wrt_parent[:3, 3]

# Extract rotation matrix
def get_joint_rotation(joint: SingleJoint, space: str = "world") -> np.ndarray:
    """Extract 3x3 rotation matrix from joint transformation."""
    if space == "world":
        return joint.joint_transform_wrt_world[:3, :3]
    else:
        return joint.joint_transform_wrt_parent[:3, :3]

# Usage
position = get_joint_position(child_joint, "world")
rotation = get_joint_rotation(child_joint, "parent")
```

### Joint Hierarchy Relationships

```python
# Check if joint is root
def is_root_joint(joint: SingleJoint) -> bool:
    """Check if joint is the root of the skeleton."""
    return joint.parent_joint_name is None

# Get joint hierarchy level (simplified)
def get_joint_level(joint: SingleJoint, all_joints: dict) -> int:
    """Get the hierarchical level of a joint (root = 0)."""
    if is_root_joint(joint):
        return 0
    
    parent = all_joints[joint.parent_joint_name]
    return 1 + get_joint_level(parent, all_joints)
```

## Coordinate Space Relationships

### Local vs. World Coordinates

```python
# For a joint hierarchy: Root -> Spine -> Chest -> LeftShoulder
# The world transform of LeftShoulder is computed as:
# world_transform = root_world @ spine_local @ chest_local @ shoulder_local

def compute_world_transform(joint: SingleJoint, parent_world: np.ndarray) -> np.ndarray:
    """Compute world transform from parent world transform and local transform."""
    return parent_world @ joint.joint_transform_wrt_parent
```

### Transform Chain Example

```python
# Building a simple arm chain
joints = {}

# Root joint (Hips)
joints["Hips"] = SingleJoint(
    joint_name="Hips",
    parent_joint_name=None,
    joint_transform_wrt_world=np.eye(4)
)

# Spine offset upward
spine_local = np.eye(4)
spine_local[1, 3] = 0.3  # 0.3 units up
joints["Spine"] = SingleJoint(
    joint_name="Spine",
    parent_joint_name="Hips",
    joint_transform_wrt_parent=spine_local,
    joint_transform_wrt_world=joints["Hips"].joint_transform_wrt_world @ spine_local
)

# Shoulder offset to the side
shoulder_local = np.eye(4)
shoulder_local[0, 3] = 0.4  # 0.4 units to the side
joints["LeftShoulder"] = SingleJoint(
    joint_name="LeftShoulder",
    parent_joint_name="Spine",
    joint_transform_wrt_parent=shoulder_local,
    joint_transform_wrt_world=joints["Spine"].joint_transform_wrt_world @ shoulder_local
)
```

## Integration Patterns

### With BVHData/BVHFlatData

`SingleJoint` objects are not directly used in the main BVH classes, but understanding their structure helps when working with the transformation data:

```python
# BVHData stores similar information per joint per frame
bvh_data = BVHData.from_file("motion.bvh")
pose = bvh_data.get_pose(0)

# Extract joint information similar to SingleJoint
joint_name = "LeftShoulder"
if joint_name in pose.joint_transforms_wrt_world:
    world_transform = pose.joint_transforms_wrt_world[joint_name]
    parent_transform = pose.joint_transforms_wrt_parent[joint_name]
    
    # Could create equivalent SingleJoint
    equivalent_joint = SingleJoint(
        joint_name=joint_name,
        joint_transform_wrt_world=world_transform,
        joint_transform_wrt_parent=parent_transform
    )
```

### Visualization Integration

```python
def visualize_single_joint(joint: SingleJoint, plotter=None):
    """Visualize a single joint with coordinate axes."""
    import igpy.myplot.vistaplot as vplot
    
    if plotter is None:
        plotter = vplot.ExPlotter.init_with_background_plotter()
    
    # Extract position and rotation
    position = joint.joint_transform_wrt_world[:3, 3]
    rotation = joint.joint_transform_wrt_world[:3, :3]
    
    # Add coordinate axes at joint position
    plotter.add_axes_many(
        position.reshape(1, 3), 
        rotation.reshape(1, 3, 3),
        axis_length=0.1
    )
    
    # Add joint name label
    plotter.add_text(
        text_content=joint.joint_name,
        position=position,
        color3f=(0, 0, 0)
    )
    
    return plotter
```

## Best Practices

!!! tip "Transformation Matrix Handling"
    - Always ensure transformation matrices are valid 4x4 homogeneous matrices
    - Verify that the bottom row is `[0, 0, 0, 1]`
    - Use `np.eye(4)` as the default for identity transformations

!!! warning "Parent-Child Consistency"
    - Ensure `parent_joint_name` references exist in the skeleton
    - Root joints must have `parent_joint_name=None`
    - Avoid circular parent-child relationships

!!! info "Memory Considerations"
    - Each `SingleJoint` stores two 4x4 matrices (128 bytes for transforms)
    - Consider using `BVHFlatData` for large numbers of joints across many frames
    - Use this class primarily for individual joint analysis or small hierarchies

## Common Pitfalls

### Invalid Transformation Matrices
```python
# ❌ Invalid - not a valid homogeneous matrix
bad_transform = np.zeros((4, 4))

# ✅ Valid - proper identity matrix
good_transform = np.eye(4)

# ✅ Valid - proper transformation with translation
good_transform = np.eye(4)
good_transform[:3, 3] = [1.0, 2.0, 3.0]  # Set translation
```

### Inconsistent Coordinate Spaces
```python
# ❌ Problematic - mixing coordinate spaces incorrectly
joint = SingleJoint(
    joint_name="Test",
    joint_transform_wrt_parent=world_space_matrix,  # Wrong space!
    joint_transform_wrt_world=parent_space_matrix   # Wrong space!
)

# ✅ Correct - consistent coordinate spaces
joint = SingleJoint(
    joint_name="Test",
    joint_transform_wrt_parent=parent_relative_matrix,
    joint_transform_wrt_world=world_space_matrix
)
```