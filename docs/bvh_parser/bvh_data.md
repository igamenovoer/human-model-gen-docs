# BVHData Class

The `BVHData` class is the primary interface for loading, manipulating, and visualizing BVH (Biovision Hierarchy) motion capture files. It provides a complete representation of skeletal animation data with frame-by-frame pose storage.

## Class Definition

```python
@define(kw_only=True, eq=False)
class BVHData:
    """Holds parsed BVH data, including the skeleton and animation."""
```

!!! info "Implementation Notes"
    - Uses the `attrs` library with `@define` decorator
    - Keyword-only initialization (`kw_only=True`)
    - Equality comparison disabled (`eq=False`)

## Attributes

### Core Properties

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `skeleton` | `nx.DiGraph` | `nx.DiGraph()` | Skeleton structure as directed graph |
| `motion_data` | `List[SinglePose]` | `[]` | List of poses for each frame |
| `source_frame_time` | `float` | `0.0` | Time duration of single frame in seconds |
| `world_transform` | `np.ndarray` | `np.eye(4)` | Global transformation matrix |

### Detailed Attribute Documentation

#### `skeleton: nx.DiGraph`
A NetworkX directed graph representing the skeletal hierarchy. Nodes are joint names, and edges represent parent-child relationships (from parent to child). Each edge stores bone length information.

```python
# Example skeleton structure
skeleton = nx.DiGraph()
skeleton.add_node("Hips")
skeleton.add_node("Spine")
skeleton.add_edge("Hips", "Spine", bone_length=0.3)  # 0.3 units long
```

#### `motion_data: List[SinglePose]`
A list of `SinglePose` objects, one for each frame of the animation. Each pose contains the complete transformation data for all joints at that specific frame.

```python
# Access frame data
frame_0_pose = bvh_data.motion_data[0]  # First frame
frame_100_pose = bvh_data.motion_data[100]  # Frame 100
```

#### `source_frame_time: float`
The time duration of a single frame in seconds, as specified in the original BVH file. Commonly 1/30 (30 fps) or 1/60 (60 fps).

```python
# Calculate animation duration
total_duration = bvh_data.n_frames * bvh_data.source_frame_time
fps = 1.0 / bvh_data.source_frame_time
```

#### `world_transform: np.ndarray`
A 4x4 transformation matrix applied to the entire skeleton in world space. Useful for repositioning or reorienting the entire animation.

```python
# Apply 90-degree rotation around Y-axis
rotation_y = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1]
])
bvh_data.world_transform = rotation_y
```

## Properties

### `root_name: str`
Get the name of the root joint (joint with no parent).

```python
root_joint = bvh_data.root_name
print(f"Root joint: {root_joint}")  # Usually "Hips" or "Root"
```

### `n_frames: int`
Get the total number of frames in the animation.

```python
total_frames = bvh_data.n_frames
print(f"Animation has {total_frames} frames")
```

## Class Methods

### `from_file(filepath: str) -> BVHData`

Load a BVH file and parse its data.

#### Parameters
- `filepath` (`str`): Path to the BVH file

#### Returns
- `BVHData`: Instance containing the parsed data

#### Example Usage

```python
from hmodel_gen.bvh_parser import BVHData

# Load BVH file
bvh_data = BVHData.from_file("path/to/motion.bvh")

# Basic information
print(f"Loaded {bvh_data.n_frames} frames")
print(f"Frame rate: {1/bvh_data.source_frame_time:.1f} fps")
print(f"Root joint: {bvh_data.root_name}")
```

## Instance Methods

### `get_pose(frame_index: int) -> SinglePose`

Get the pose at a specific frame index with world transform applied.

#### Parameters
- `frame_index` (`int`): Index of the frame to retrieve

#### Returns
- `SinglePose`: Pose data for the specified frame

#### Example Usage

```python
# Get pose at frame 50
pose = bvh_data.get_pose(50)

# Get joint positions
positions = pose.get_joint_positions("world")
print(f"Hip position: {positions['Hips']}")
```

### `visualize_pose(frame_index: int, plotter=None, show_axes=True, show_names=False)`

Visualize the pose at a specific frame index.

#### Parameters
- `frame_index` (`int`): Index of the frame to visualize
- `plotter` (`vplot.ExPlotter`, optional): Existing plotter instance
- `show_axes` (`bool`, optional): Display axes for each joint. Default: `True`
- `show_names` (`bool`, optional): Display joint names. Default: `False`

#### Returns
- `vplot.ExPlotter`: Plotter instance used for visualization

#### Example Usage

```python
# Basic visualization
plotter = bvh_data.visualize_pose(0)

# Detailed visualization with names
plotter = bvh_data.visualize_pose(
    frame_index=100,
    show_axes=True,
    show_names=True
)

# Use existing plotter
import igpy.myplot.vistaplot as vplot
my_plotter = vplot.ExPlotter.init_with_background_plotter()
bvh_data.visualize_pose(0, plotter=my_plotter)
```

### `clone() -> BVHData`

Create a deep copy of the BVHData instance.

#### Returns
- `BVHData`: New instance that is a deep copy

#### Example Usage

```python
# Create independent copy
bvh_copy = bvh_data.clone()

# Modify copy without affecting original
bvh_copy.world_transform = rotation_matrix
```

### `get_sub_skeleton(joint_names: List[str]) -> BVHData`

Create a new BVHData object with a subset of joints.

#### Parameters
- `joint_names` (`List[str]`): Joint names to include in sub-skeleton

#### Returns
- `BVHData`: New instance with specified sub-skeleton

#### Raises
- `ValueError`: If resulting sub-skeleton doesn't have a single root

#### Example Usage

```python
# Extract upper body
upper_body_joints = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand"
]

upper_body_data = bvh_data.get_sub_skeleton(upper_body_joints)
```

### `get_bone_lengths() -> Dict[tuple[str, str], float]`

Get the lengths of all bones in the skeleton.

#### Returns
- `Dict[tuple[str, str], float]`: Dictionary mapping (parent, child) to bone length

#### Example Usage

```python
bone_lengths = bvh_data.get_bone_lengths()

# Print all bone lengths
for (parent, child), length in bone_lengths.items():
    print(f"{parent} -> {child}: {length:.3f}")

# Get specific bone length
spine_length = bone_lengths[("Hips", "Spine")]
```

### `update_bone_lengths(bone_lengths: Dict[tuple[str, str], float])`

Update bone lengths and recompute all pose transformations.

#### Parameters
- `bone_lengths` (`Dict[tuple[str, str], float]`): New bone lengths to apply

#### Description
This method preserves joint rotations while updating translations to match new bone lengths. All pose transformations are recomputed to maintain consistency.

#### Example Usage

```python
# Get current bone lengths
current_lengths = bvh_data.get_bone_lengths()

# Scale all bones by 1.2x
scaled_lengths = {bone: length * 1.2 for bone, length in current_lengths.items()}

# Apply new lengths
bvh_data.update_bone_lengths(scaled_lengths)

# Or modify specific bones
new_lengths = current_lengths.copy()
new_lengths[("Hips", "LeftUpLeg")] *= 1.5  # Make left thigh 50% longer
bvh_data.update_bone_lengths(new_lengths)
```

## Magic Methods

### `__len__() -> int`
Get the number of frames in the animation.

```python
num_frames = len(bvh_data)
```

### `__getitem__(key) -> Union[BVHData, SinglePose]`
Support indexing and slicing operations.

#### Parameters
- `key`: Integer index or slice object

#### Returns
- `SinglePose`: If key is an integer
- `BVHData`: If key is a slice

#### Example Usage

```python
# Get single frame
pose = bvh_data[50]  # Returns SinglePose

# Get frame range
segment = bvh_data[10:20]  # Returns BVHData with frames 10-19
segment = bvh_data[::2]    # Returns BVHData with every other frame

# Get last frame
last_pose = bvh_data[-1]
```

## Comprehensive Usage Examples

### Basic BVH Loading and Analysis

```python
from hmodel_gen.bvh_parser import BVHData
import numpy as np

# Load BVH file
bvh_data = BVHData.from_file("dance_motion.bvh")

# Basic statistics
print(f"Animation: {bvh_data.n_frames} frames at {1/bvh_data.source_frame_time:.1f} fps")
print(f"Duration: {bvh_data.n_frames * bvh_data.source_frame_time:.2f} seconds")
print(f"Root joint: {bvh_data.root_name}")
print(f"Joints: {list(bvh_data.skeleton.nodes())}")

# Analyze joint hierarchy
print("\nJoint hierarchy:")
for parent, child in bvh_data.skeleton.edges():
    bone_length = bvh_data.skeleton.edges[parent, child]['bone_length']
    print(f"  {parent} -> {child}: {bone_length:.3f} units")
```

### Motion Analysis

```python
def analyze_motion_range(bvh_data: BVHData, joint_name: str):
    """Analyze the range of motion for a specific joint."""
    positions = []
    
    for frame_idx in range(bvh_data.n_frames):
        pose = bvh_data.get_pose(frame_idx)
        pos = pose.get_joint_positions("world")[joint_name]
        positions.append(pos)
    
    positions = np.array(positions)
    
    # Calculate statistics
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    range_pos = max_pos - min_pos
    mean_pos = np.mean(positions, axis=0)
    
    print(f"Joint: {joint_name}")
    print(f"  Range: X={range_pos[0]:.3f}, Y={range_pos[1]:.3f}, Z={range_pos[2]:.3f}")
    print(f"  Center: X={mean_pos[0]:.3f}, Y={mean_pos[1]:.3f}, Z={mean_pos[2]:.3f}")
    
    return {
        'positions': positions,
        'range': range_pos,
        'center': mean_pos,
        'min': min_pos,
        'max': max_pos
    }

# Analyze hip motion
hip_analysis = analyze_motion_range(bvh_data, "Hips")
```

### Animation Retargeting

```python
def retarget_animation(source_bvh: BVHData, target_bone_lengths: Dict[tuple[str, str], float]) -> BVHData:
    """Retarget animation to different bone lengths."""
    # Clone the source
    retargeted = source_bvh.clone()
    
    # Update bone lengths
    retargeted.update_bone_lengths(target_bone_lengths)
    
    return retargeted

# Load source animation
source = BVHData.from_file("source_motion.bvh")

# Define target skeleton proportions
target_lengths = source.get_bone_lengths()
target_lengths[("Hips", "LeftUpLeg")] *= 1.3  # Longer legs
target_lengths[("Hips", "RightUpLeg")] *= 1.3
target_lengths[("LeftUpLeg", "LeftLeg")] *= 1.3
target_lengths[("RightUpLeg", "RightLeg")] *= 1.3

# Create retargeted animation
retargeted = retarget_animation(source, target_lengths)
```

### Animation Editing

```python
def create_animation_loop(bvh_data: BVHData, start_frame: int, end_frame: int) -> BVHData:
    """Create a looping animation segment."""
    if start_frame >= end_frame or end_frame >= bvh_data.n_frames:
        raise ValueError("Invalid frame range")
    
    # Extract segment
    segment = bvh_data[start_frame:end_frame]
    
    # Optionally blend first and last frames for smooth looping
    # (This is simplified - real blending would be more sophisticated)
    
    return segment

def concatenate_animations(*bvh_datas: BVHData) -> BVHData:
    """Concatenate multiple BVH animations."""
    if not bvh_datas:
        raise ValueError("No animations provided")
    
    # Verify all animations have same skeleton structure
    first_skeleton = bvh_datas[0].skeleton
    for bvh in bvh_datas[1:]:
        if set(bvh.skeleton.nodes()) != set(first_skeleton.nodes()):
            raise ValueError("All animations must have the same skeleton structure")
    
    # Combine motion data
    combined_motion = []
    for bvh in bvh_datas:
        combined_motion.extend(bvh.motion_data)
    
    # Create new BVHData
    result = BVHData(
        skeleton=first_skeleton.copy(),
        motion_data=combined_motion,
        source_frame_time=bvh_datas[0].source_frame_time,
        world_transform=bvh_datas[0].world_transform.copy()
    )
    
    return result

# Usage
loop_segment = create_animation_loop(bvh_data, 10, 50)
extended_animation = concatenate_animations(bvh_data, loop_segment, bvh_data)
```

### Visualization Workflows

```python
def create_animation_preview(bvh_data: BVHData, frame_indices: List[int]):
    """Create a multi-frame preview visualization."""
    import igpy.myplot.vistaplot as vplot
    
    plotter = vplot.ExPlotter.init_with_background_plotter()
    
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]  # Red, Green, Blue, Yellow, Magenta
    
    for i, frame_idx in enumerate(frame_indices):
        pose = bvh_data.get_pose(frame_idx)
        positions = pose.get_joint_positions("world")
        
        # Plot joints
        points = np.array(list(positions.values()))
        color = colors[i % len(colors)]
        plotter.add_point_cloud(points, color3f=color)
        
        # Plot bones
        pts1, pts2 = [], []
        for joint_name, position in positions.items():
            for child_name in bvh_data.skeleton.successors(joint_name):
                child_pos = positions[child_name]
                pts1.append(position)
                pts2.append(child_pos)
        
        if pts1:
            plotter.add_line_segments(
                np.array(pts1), np.array(pts2), 
                color3f=color, line_width=1.0
            )
    
    return plotter

# Create preview of keyframes
keyframes = [0, 25, 50, 75, 100]
preview_plotter = create_animation_preview(bvh_data, keyframes)
```

## Performance Considerations

### Memory Usage

```python
def estimate_memory_usage(bvh_data: BVHData) -> dict:
    """Estimate memory usage of BVHData object."""
    n_joints = len(bvh_data.skeleton.nodes())
    n_frames = bvh_data.n_frames
    
    # Each transformation matrix: 4x4 floats = 64 bytes
    # Each pose has 2 transform dicts (world + parent) = 128 bytes per joint per frame
    pose_data_size = n_joints * n_frames * 128
    
    # Skeleton graph overhead (approximate)
    skeleton_size = n_joints * 100  # Rough estimate
    
    total_mb = (pose_data_size + skeleton_size) / (1024 * 1024)
    
    return {
        'joints': n_joints,
        'frames': n_frames,
        'pose_data_mb': pose_data_size / (1024 * 1024),
        'skeleton_mb': skeleton_size / (1024 * 1024),
        'total_mb': total_mb
    }

# Check memory usage
memory_info = estimate_memory_usage(bvh_data)
print(f"Estimated memory usage: {memory_info['total_mb']:.2f} MB")
```

### When to Use BVHFlatData

```python
# Rule of thumb: Use BVHFlatData for large datasets
def should_use_flat_data(n_frames: int, n_joints: int) -> bool:
    """Determine if BVHFlatData would be more efficient."""
    # Rough threshold based on memory usage
    estimated_mb = (n_frames * n_joints * 128) / (1024 * 1024)
    return estimated_mb > 50  # Switch to flat data above 50 MB

if should_use_flat_data(bvh_data.n_frames, len(bvh_data.skeleton.nodes())):
    print("Consider using BVHFlatData for better performance")
    # Convert to flat data
    from hmodel_gen.bvh_parser import BVHFlatData
    flat_data = BVHFlatData.from_file("large_motion.bvh")
```

## Best Practices

!!! tip "Loading and Validation"
    - Always validate BVH files before processing
    - Check frame count and joint hierarchy after loading
    - Verify bone lengths are reasonable for your application

!!! warning "Memory Management"
    - Monitor memory usage for large animations
    - Use `BVHFlatData` for animations with >5000 frames
    - Consider sub-skeleton extraction for memory reduction

!!! info "Coordinate Spaces"
    - Use world space for visualization and absolute positioning
    - Use parent-relative space for animation editing and retargeting
    - Apply world transforms through `world_transform` attribute, not manual modification

## Common Errors and Solutions

### File Loading Issues
```python
try:
    bvh_data = BVHData.from_file("motion.bvh")
except FileNotFoundError:
    print("BVH file not found")
except Exception as e:
    print(f"Error loading BVH file: {e}")
```

### Invalid Frame Access
```python
# ❌ Can cause IndexError
frame_index = 1000
pose = bvh_data.get_pose(frame_index)

# ✅ Safe access
if 0 <= frame_index < bvh_data.n_frames:
    pose = bvh_data.get_pose(frame_index)
else:
    print(f"Frame {frame_index} out of range [0, {bvh_data.n_frames-1}]")
```

### Sub-skeleton Validation
```python
# ❌ Can cause ValueError if joints don't form connected hierarchy
try:
    sub_skeleton = bvh_data.get_sub_skeleton(["LeftHand", "RightFoot"])  # Disconnected
except ValueError as e:
    print(f"Sub-skeleton error: {e}")

# ✅ Ensure connected hierarchy
upper_body = ["Hips", "Spine", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
sub_skeleton = bvh_data.get_sub_skeleton(upper_body)
```