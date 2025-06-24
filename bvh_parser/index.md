# BVH Parser Module

The BVH Parser module provides comprehensive tools for loading, manipulating, and visualizing BVH (Biovision Hierarchy) motion capture data. This module is designed to handle skeletal animation data efficiently while maintaining full compatibility with standard BVH file formats.

## Module Overview

The BVH parser consists of four main classes that work together to provide a complete motion capture data processing pipeline:

### Core Classes

| Class | Purpose | Use Case |
|-------|---------|----------|
| [**SingleJoint**](single_joint.md) | Individual joint representation | Storing joint-specific transformation data |
| [**SinglePose**](single_pose.md) | Complete pose for a single frame | Frame-specific pose queries and transformations |
| [**BVHData**](bvh_data.md) | Standard BVH data structure | General-purpose BVH file loading and manipulation |
| [**BVHFlatData**](bvh_flat_data.md) | Memory-efficient storage | Large animation sequences with optimized memory usage |

## Key Features

### 🔄 Coordinate Space Management
- **World Space**: Global coordinate transformations
- **Parent-Relative Space**: Local joint transformations
- **Automatic Conversion**: Seamless switching between coordinate systems

### 📊 Data Structures
- **Hierarchical Storage**: NetworkX-based skeleton graphs
- **Efficient Access**: Frame-indexed pose retrieval
- **Memory Optimization**: Flat data structure for large sequences

### 🎨 Visualization
- **3D Rendering**: Interactive skeleton visualization
- **Joint Axes**: Optional coordinate frame display
- **Bone Connections**: Visual representation of skeletal hierarchy
- **Text Labels**: Joint name annotations

### 🔧 Manipulation Tools
- **Pose Extraction**: Individual frame access
- **Sub-skeleton Creation**: Extract specific joint subsets
- **Bone Length Modification**: Dynamic skeleton scaling
- **Transform Application**: Global transformations to entire sequences

## Architecture Pattern

```mermaid
graph TB
    A[BVH File] --> B[BVHData.from_file()]
    B --> C[NetworkX Skeleton Graph]
    B --> D[Motion Data List]
    
    C --> E[Joint Hierarchy]
    D --> F[Frame-by-Frame Poses]
    
    E --> G[SingleJoint Objects]
    F --> H[SinglePose Objects]
    
    G --> I[Joint Transformations]
    H --> J[Pose Transformations]
    
    I --> K[Visualization]
    J --> K
    
    B --> L[BVHFlatData Conversion]
    L --> M[Memory-Efficient Storage]
```

## Usage Patterns

### Basic BVH Loading
```python
from hmodel_gen.bvh_parser import BVHData

# Load BVH file
bvh_data = BVHData.from_file("motion.bvh")

# Get basic information
print(f"Animation has {bvh_data.n_frames} frames")
print(f"Root joint: {bvh_data.root_name}")
print(f"Frame time: {bvh_data.source_frame_time}s")

# Access specific frame
pose = bvh_data.get_pose(100)
```

### Memory-Efficient Processing
```python
from hmodel_gen.bvh_parser import BVHFlatData

# For large animations, use flat data structure
bvh_flat = BVHFlatData.from_file("long_animation.bvh")

# Same interface, better memory performance
pose = bvh_flat.get_pose(1000)
```

### Visualization
```python
# Visualize a specific frame
plotter = bvh_data.visualize_pose(
    frame_index=0,
    show_axes=True,      # Show joint coordinate frames
    show_names=True      # Show joint names
)
```

### Sub-skeleton Extraction
```python
# Extract upper body joints
upper_body_joints = [
    "Hips", "Spine", "Spine1", "Spine2", 
    "LeftShoulder", "LeftArm", "LeftForeArm",
    "RightShoulder", "RightArm", "RightForeArm"
]

upper_body_data = bvh_data.get_sub_skeleton(upper_body_joints)
```

### Bone Length Manipulation
```python
# Get current bone lengths
bone_lengths = bvh_data.get_bone_lengths()

# Modify specific bones
new_lengths = bone_lengths.copy()
new_lengths[("Hips", "LeftUpLeg")] *= 1.2  # Make left leg 20% longer

# Apply changes
bvh_data.update_bone_lengths(new_lengths)
```

## Data Flow

The BVH parser follows a clear data flow pattern:

1. **File Loading**: BVH files are parsed using the `bvhsdk` library
2. **Skeleton Construction**: Joint hierarchy is built as a NetworkX directed graph
3. **Motion Data**: Frame-by-frame transformations are stored
4. **Pose Access**: Individual frames can be retrieved and manipulated
5. **Visualization**: 3D rendering through the `igpy` visualization library

## Coordinate Systems

### World Space Transformations
- All joints positioned relative to global origin
- Useful for absolute positioning and visualization
- Automatically computed from local transformations

### Parent-Relative Transformations
- Joints positioned relative to their parent joint
- Maintains original BVH file structure
- Essential for animation retargeting and modification

## Performance Considerations

### BVHData vs BVHFlatData

| Aspect | BVHData | BVHFlatData |
|--------|---------|-------------|
| **Memory Usage** | Higher (list of poses) | Lower (numpy arrays) |
| **Access Speed** | Slower for long sequences | Faster for bulk operations |
| **Use Case** | General purpose, small-medium sequences | Large animations, batch processing |
| **Flexibility** | Full pose object access | Optimized for specific operations |

### Best Practices

!!! tip "Choose the Right Data Structure"
    - Use `BVHData` for general-purpose work and sequences under 1000 frames
    - Use `BVHFlatData` for large animations or when memory is constrained
    - Both classes provide identical interfaces for most operations

!!! warning "Memory Considerations"
    - Long BVH sequences can consume significant memory
    - Consider using `BVHFlatData` for sequences over 5000 frames
    - Sub-skeleton extraction can reduce memory usage for specific applications

!!! info "Visualization Performance"
    - Background plotters display automatically without blocking
    - Standard plotters require explicit `show()` calls
    - Large skeletons benefit from selective joint visualization

## Integration with Other Modules

The BVH parser integrates seamlessly with other parts of the HumanModelGeneration system:

- **Blender Export**: Compatible with skeleton data exported from Blender
- **SMPL Integration**: Provides motion data for SMPL-X model driving
- **Visualization**: Uses `igpy.myplot.vistaplot` for 3D rendering
- **Analysis**: Supports motion analysis and pose comparison workflows