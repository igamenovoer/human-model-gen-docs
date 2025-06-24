# BVHFlatData Class

The `BVHFlatData` class provides a memory-efficient alternative to `BVHData` for handling large BVH animations. Instead of storing a list of pose objects, it uses large NumPy arrays in a "transposed" layout, significantly reducing memory overhead for long sequences.

## Class Definition

```python
@define(kw_only=True, eq=False)
class BVHFlatData:
    """Holds parsed BVH data in a memory-efficient, 'transposed' layout."""
```

!!! info "Implementation Notes"
    - Uses the `attrs` library with `@define` decorator
    - Keyword-only initialization (`kw_only=True`)
    - Equality comparison disabled (`eq=False`)
    - Optimized for large animations (>1000 frames)

## Attributes

### Core Properties

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `skeleton` | `nx.DiGraph` | `nx.DiGraph()` | Skeleton structure as directed graph |
| `joint_transforms_wrt_parent` | `Dict[str, np.ndarray]` | `{}` | (N,4,4) parent-relative transforms per joint |
| `joint_transforms_wrt_world` | `Dict[str, np.ndarray]` | `{}` | (N,4,4) world-space transforms per joint |
| `source_frame_index` | `np.ndarray` | `np.array([])` | (N,) array of frame indices |
| `source_frame_time` | `float` | `0.0` | Time duration of single frame in seconds |
| `world_transform` | `np.ndarray` | `np.eye(4)` | Global transformation matrix |

### Detailed Attribute Documentation

#### `skeleton: nx.DiGraph`
Identical to `BVHData.skeleton` - a NetworkX directed graph representing the skeletal hierarchy with bone length information stored on edges.

```python
# Same structure as BVHData
for parent, child in flat_data.skeleton.edges():
    bone_length = flat_data.skeleton.edges[parent, child]['bone_length']
    print(f"{parent} -> {child}: {bone_length:.3f}")
```

#### `joint_transforms_wrt_parent: Dict[str, np.ndarray]`
Dictionary mapping joint names to (N, 4, 4) NumPy arrays where N is the number of frames. Each array contains the transformation matrices for that joint across all frames, relative to its parent.

```python
# Shape: (n_frames, 4, 4)
spine_transforms = flat_data.joint_transforms_wrt_parent["Spine"]
print(f"Spine transforms shape: {spine_transforms.shape}")

# Access frame 100's transform for Spine joint
frame_100_transform = spine_transforms[100]  # Shape: (4, 4)
```

#### `joint_transforms_wrt_world: Dict[str, np.ndarray]`
Dictionary mapping joint names to (N, 4, 4) NumPy arrays containing world-space transformations for each joint across all frames.

```python
# All world transforms for left shoulder across animation
left_shoulder_world = flat_data.joint_transforms_wrt_world["LeftShoulder"]

# Extract positions across all frames
positions = left_shoulder_world[:, :3, 3]  # Shape: (n_frames, 3)
```

#### `source_frame_index: np.ndarray`
An (N,) array containing the original frame indices from the motion sequence. Useful for tracking frame correspondence after slicing operations.

```python
# Original frame indices
print(f"Frame indices: {flat_data.source_frame_index[:10]}")  # First 10 frames

# After slicing, indices are preserved
subset = flat_data[10:20]
print(f"Subset indices: {subset.source_frame_index}")  # [10, 11, 12, ..., 19]
```

## Properties

### `root_name: str`
Get the name of the root joint (identical to `BVHData`).

```python
root_joint = flat_data.root_name
```

### `n_frames: int`
Get the total number of frames in the animation.

```python
total_frames = flat_data.n_frames
```

## Class Methods

### `from_file(filepath: str) -> BVHFlatData`

Load a BVH file and parse its data into the flat structure.

#### Parameters
- `filepath` (`str`): Path to the BVH file

#### Returns
- `BVHFlatData`: Instance containing the parsed data in flat format

#### Example Usage

```python
from hmodel_gen.bvh_parser import BVHFlatData

# Load large BVH file efficiently
flat_data = BVHFlatData.from_file("large_animation.bvh")

print(f"Loaded {flat_data.n_frames} frames efficiently")
print(f"Memory layout: {flat_data.joint_transforms_wrt_world['Hips'].shape}")
```

## Instance Methods

### `get_pose(frame_index: int) -> SinglePose`

Construct and return a SinglePose object for a specific frame.

#### Parameters
- `frame_index` (`int`): Index of the frame to retrieve

#### Returns
- `SinglePose`: Pose data for the specified frame

#### Raises
- `IndexError`: If frame index is out of range

#### Example Usage

```python
# Get pose at frame 500
pose = flat_data.get_pose(500)

# Same interface as BVHData
positions = pose.get_joint_positions("world")
```

### `visualize_pose(frame_index: int, plotter=None, show_axes=True, show_names=False)`

Visualize the pose at a specific frame index (identical interface to `BVHData`).

#### Parameters
- `frame_index` (`int`): Index of the frame to visualize
- `plotter` (`vplot.ExPlotter`, optional): Existing plotter instance
- `show_axes` (`bool`, optional): Display axes for each joint. Default: `True`
- `show_names` (`bool`, optional): Display joint names. Default: `False`

#### Returns
- `vplot.ExPlotter`: Plotter instance used for visualization

### `clone() -> BVHFlatData`

Create a deep copy of the BVHFlatData instance.

#### Returns
- `BVHFlatData`: New instance that is a deep copy

### `get_sub_skeleton(joint_names: List[str]) -> BVHFlatData`

Create a new BVHFlatData object with a subset of joints.

#### Parameters
- `joint_names` (`List[str]`): Joint names to include in sub-skeleton

#### Returns
- `BVHFlatData`: New instance with specified sub-skeleton

#### Raises
- `ValueError`: If resulting sub-skeleton doesn't have a single root

### `get_bone_lengths() -> Dict[tuple[str, str], float]`

Get the lengths of all bones in the skeleton (identical to `BVHData`).

### `update_bone_lengths(bone_lengths: Dict[tuple[str, str], float])`

Update bone lengths and recompute all pose transformations. This method is optimized for the flat data structure, processing all frames simultaneously using vectorized operations.

#### Parameters
- `bone_lengths` (`Dict[tuple[str, str], float]`): New bone lengths to apply

#### Example Usage

```python
# Get current lengths
current_lengths = flat_data.get_bone_lengths()

# Scale specific bones
new_lengths = current_lengths.copy()
new_lengths[("Hips", "Spine")] *= 1.2

# Update efficiently across all frames
flat_data.update_bone_lengths(new_lengths)
```

## Magic Methods

### `__len__() -> int`
Get the number of frames in the animation.

### `__getitem__(key) -> Union[BVHFlatData, SinglePose]`
Support indexing and slicing operations with preserved efficiency.

#### Parameters
- `key`: Integer index or slice object

#### Returns
- `SinglePose`: If key is an integer
- `BVHFlatData`: If key is a slice (maintains flat structure)

#### Example Usage

```python
# Get single frame (constructs SinglePose)
pose = flat_data[100]

# Get frame range (maintains flat structure)
segment = flat_data[1000:2000]  # Still BVHFlatData
print(f"Segment has {segment.n_frames} frames")

# Slicing with step
every_10th = flat_data[::10]  # Every 10th frame
```

## Memory Efficiency Analysis

### Memory Layout Comparison

| Data Structure | Memory Pattern | Efficiency for Large Sequences |
|----------------|----------------|-------------------------------|
| `BVHData` | List of pose objects | Lower (object overhead) |
| `BVHFlatData` | Contiguous NumPy arrays | Higher (vectorized operations) |

### Memory Usage Example

```python
def compare_memory_usage(filepath: str):
    """Compare memory usage between BVHData and BVHFlatData."""
    import sys
    from hmodel_gen.bvh_parser import BVHData, BVHFlatData
    
    # Load with BVHData
    bvh_data = BVHData.from_file(filepath)
    bvh_size = sys.getsizeof(bvh_data.motion_data)
    
    # Load with BVHFlatData
    flat_data = BVHFlatData.from_file(filepath)
    flat_size = sum(sys.getsizeof(arr) for arr in flat_data.joint_transforms_wrt_world.values())
    
    print(f"BVHData motion data: ~{bvh_size / 1024 / 1024:.2f} MB")
    print(f"BVHFlatData arrays: ~{flat_size / 1024 / 1024:.2f} MB")
    print(f"Memory efficiency: {bvh_size / flat_size:.2f}x")

# Usage
compare_memory_usage("large_motion.bvh")
```

## Performance Optimization Examples

### Batch Processing

```python
def analyze_joint_trajectory(flat_data: BVHFlatData, joint_name: str):
    """Analyze joint trajectory across entire animation efficiently."""
    # Get all transforms for the joint at once
    transforms = flat_data.joint_transforms_wrt_world[joint_name]  # Shape: (n_frames, 4, 4)
    
    # Extract positions for all frames simultaneously
    positions = transforms[:, :3, 3]  # Shape: (n_frames, 3)
    
    # Vectorized analysis
    velocity = np.diff(positions, axis=0)  # Frame-to-frame velocity
    speed = np.linalg.norm(velocity, axis=1)  # Speed magnitude
    
    # Statistics
    max_speed = np.max(speed)
    avg_speed = np.mean(speed)
    total_distance = np.sum(np.linalg.norm(velocity, axis=1))
    
    return {
        'positions': positions,
        'velocity': velocity,
        'speed': speed,
        'max_speed': max_speed,
        'avg_speed': avg_speed,
        'total_distance': total_distance
    }

# Efficient analysis
hip_analysis = analyze_joint_trajectory(flat_data, "Hips")
print(f"Hip max speed: {hip_analysis['max_speed']:.3f}")
```

### Vectorized Transformations

```python
def apply_global_scale(flat_data: BVHFlatData, scale_factor: float):
    """Apply scaling to all joint positions across all frames."""
    # Create scale matrix
    scale_matrix = np.eye(4) * scale_factor
    scale_matrix[3, 3] = 1.0  # Keep homogeneous coordinate
    
    # Apply to all joints and frames efficiently
    for joint_name in flat_data.joint_transforms_wrt_world:
        # Vectorized matrix multiplication across all frames
        transforms = flat_data.joint_transforms_wrt_world[joint_name]
        flat_data.joint_transforms_wrt_world[joint_name] = transforms @ scale_matrix
        
        # Update parent-relative transforms for root
        if joint_name == flat_data.root_name:
            flat_data.joint_transforms_wrt_parent[joint_name] = transforms @ scale_matrix

# Usage
scaled_data = flat_data.clone()
apply_global_scale(scaled_data, 1.2)  # Scale up by 20%
```

### Frame Range Processing

```python
def process_frame_range(flat_data: BVHFlatData, start_frame: int, end_frame: int, 
                       process_func) -> np.ndarray:
    """Process a range of frames efficiently."""
    results = []
    
    for joint_name in flat_data.skeleton.nodes():
        # Extract frame range for this joint
        joint_transforms = flat_data.joint_transforms_wrt_world[joint_name]
        frame_range = joint_transforms[start_frame:end_frame]
        
        # Apply processing function
        result = process_func(frame_range)
        results.append(result)
    
    return np.array(results)

def compute_joint_variance(transforms: np.ndarray) -> float:
    """Compute variance in joint positions."""
    positions = transforms[:, :3, 3]
    return np.var(positions)

# Find most active joints in frame range
variances = process_frame_range(flat_data, 100, 200, compute_joint_variance)
joint_names = list(flat_data.skeleton.nodes())
most_active = joint_names[np.argmax(variances)]
print(f"Most active joint: {most_active}")
```

## Advanced Usage Patterns

### Animation Blending

```python
def blend_animations(flat_data1: BVHFlatData, flat_data2: BVHFlatData, 
                    blend_weight: float) -> BVHFlatData:
    """Blend two animations with specified weight."""
    if blend_weight < 0 or blend_weight > 1:
        raise ValueError("Blend weight must be between 0 and 1")
    
    # Ensure same structure
    if (set(flat_data1.skeleton.nodes()) != set(flat_data2.skeleton.nodes()) or
        flat_data1.n_frames != flat_data2.n_frames):
        raise ValueError("Animations must have same structure and frame count")
    
    # Create result
    result = flat_data1.clone()
    
    # Blend transforms for each joint
    for joint_name in flat_data1.skeleton.nodes():
        transforms1 = flat_data1.joint_transforms_wrt_world[joint_name]
        transforms2 = flat_data2.joint_transforms_wrt_world[joint_name]
        
        # Linear blend (note: not ideal for rotations)
        blended = (1 - blend_weight) * transforms1 + blend_weight * transforms2
        result.joint_transforms_wrt_world[joint_name] = blended
        
        # Blend parent-relative transforms
        parent1 = flat_data1.joint_transforms_wrt_parent[joint_name]
        parent2 = flat_data2.joint_transforms_wrt_parent[joint_name]
        blended_parent = (1 - blend_weight) * parent1 + blend_weight * parent2
        result.joint_transforms_wrt_parent[joint_name] = blended_parent
    
    return result

# Usage
blended = blend_animations(flat_data1, flat_data2, 0.5)  # 50/50 blend
```

### Motion Segmentation

```python
def segment_by_activity(flat_data: BVHFlatData, joint_name: str = "Hips", 
                       threshold: float = 0.1) -> List[tuple[int, int]]:
    """Segment animation based on joint activity level."""
    # Get joint positions
    transforms = flat_data.joint_transforms_wrt_world[joint_name]
    positions = transforms[:, :3, 3]
    
    # Compute velocity
    velocity = np.diff(positions, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    
    # Find active segments
    active_frames = speed > threshold
    
    # Find segment boundaries
    segments = []
    start = None
    
    for i, is_active in enumerate(active_frames):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            segments.append((start, i))
            start = None
    
    # Handle case where animation ends during active segment
    if start is not None:
        segments.append((start, len(active_frames)))
    
    return segments

# Find active segments
active_segments = segment_by_activity(flat_data, "Hips", threshold=0.05)
print(f"Found {len(active_segments)} active segments:")
for start, end in active_segments:
    duration = (end - start) * flat_data.source_frame_time
    print(f"  Frames {start}-{end} ({duration:.2f}s)")
```

## Best Practices

!!! tip "When to Use BVHFlatData"
    - Animations with >1000 frames
    - Batch processing operations
    - Memory-constrained environments
    - Vectorized analysis workflows

!!! warning "Memory Allocation"
    - Initial loading may use more memory temporarily
    - NumPy arrays are allocated contiguously
    - Consider available RAM for very large animations

!!! info "Performance Tips"
    - Use vectorized operations instead of frame-by-frame loops
    - Slice operations maintain flat structure efficiency
    - Prefer batch processing over individual frame access

## Conversion Between Formats

### BVHData to BVHFlatData

```python
def convert_to_flat(bvh_data: BVHData) -> BVHFlatData:
    """Convert BVHData to BVHFlatData format."""
    n_frames = bvh_data.n_frames
    joint_names = list(bvh_data.skeleton.nodes())
    
    # Initialize arrays
    joint_transforms_wrt_parent = {
        name: np.zeros((n_frames, 4, 4)) for name in joint_names
    }
    joint_transforms_wrt_world = {
        name: np.zeros((n_frames, 4, 4)) for name in joint_names
    }
    
    # Fill arrays from pose data
    for frame_idx in range(n_frames):
        pose = bvh_data.motion_data[frame_idx]
        for joint_name in joint_names:
            joint_transforms_wrt_parent[joint_name][frame_idx] = \
                pose.joint_transforms_wrt_parent[joint_name]
            joint_transforms_wrt_world[joint_name][frame_idx] = \
                pose.joint_transforms_wrt_world[joint_name]
    
    return BVHFlatData(
        skeleton=bvh_data.skeleton.copy(),
        joint_transforms_wrt_parent=joint_transforms_wrt_parent,
        joint_transforms_wrt_world=joint_transforms_wrt_world,
        source_frame_index=np.arange(n_frames),
        source_frame_time=bvh_data.source_frame_time,
        world_transform=bvh_data.world_transform.copy()
    )
```

### BVHFlatData to BVHData

```python
def convert_from_flat(flat_data: BVHFlatData) -> BVHData:
    """Convert BVHFlatData to BVHData format."""
    motion_data = []
    
    for frame_idx in range(flat_data.n_frames):
        pose_world = {}
        pose_parent = {}
        
        for joint_name in flat_data.skeleton.nodes():
            pose_world[joint_name] = flat_data.joint_transforms_wrt_world[joint_name][frame_idx]
            pose_parent[joint_name] = flat_data.joint_transforms_wrt_parent[joint_name][frame_idx]
        
        pose = SinglePose(
            joint_transforms_wrt_world=pose_world,
            joint_transforms_wrt_parent=pose_parent,
            source_frame_index=flat_data.source_frame_index[frame_idx]
        )
        motion_data.append(pose)
    
    return BVHData(
        skeleton=flat_data.skeleton.copy(),
        motion_data=motion_data,
        source_frame_time=flat_data.source_frame_time,
        world_transform=flat_data.world_transform.copy()
    )
```

## Common Pitfalls and Solutions

### Array Shape Mismatches
```python
# ❌ Wrong - assuming 2D array
positions = flat_data.joint_transforms_wrt_world["Hips"][:3, 3]  # Error!

# ✅ Correct - 3D array with frame dimension
positions = flat_data.joint_transforms_wrt_world["Hips"][:, :3, 3]  # (n_frames, 3)
```

### Inefficient Frame Access
```python
# ❌ Inefficient - creating SinglePose for each frame
all_positions = []
for i in range(flat_data.n_frames):
    pose = flat_data.get_pose(i)
    positions = pose.get_joint_positions("world")
    all_positions.append(positions["Hips"])

# ✅ Efficient - direct array access
hip_positions = flat_data.joint_transforms_wrt_world["Hips"][:, :3, 3]
```

### Memory Management
```python
# ❌ Creating unnecessary copies
processed_data = []
for joint_name in flat_data.skeleton.nodes():
    transforms = flat_data.joint_transforms_wrt_world[joint_name].copy()  # Unnecessary copy
    processed_data.append(process_transforms(transforms))

# ✅ Process in-place or with views
processed_data = []
for joint_name in flat_data.skeleton.nodes():
    transforms = flat_data.joint_transforms_wrt_world[joint_name]  # View
    processed_data.append(process_transforms(transforms))
```