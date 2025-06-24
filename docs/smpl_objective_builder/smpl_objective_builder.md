# SmplObjectiveBuilder Class

The `SmplObjectiveBuilder` class is the core component for creating optimization objectives to control SMPL-X parametric 3D human models. It provides a comprehensive interface for configuring degrees of freedom and building PyTorch-based optimization objectives.

## Class Definition

```python
class SmplObjectiveBuilder:
    """Builds optimization objectives to control the SMPLx parametric 3D human model."""
```

!!! info "Design Philosophy"
    - Member variables use `m_` prefix following project conventions
    - Properties provide read-only access to internal state
    - Setters use `set_var_xxx()` pattern for mutable properties
    - All objectives return PyTorch tensors with gradient tracking enabled

## Attributes

### Core Properties

| Attribute | Type | Description |
|-----------|------|-------------|
| `m_model` | `smplx.SMPLX \| None` | The SMPL-X model instance |
| `m_dof_translation` | `dict` | Translation DoF flags `{'x': bool, 'y': bool, 'z': bool}` |
| `m_dof_rotation` | `dict` | Rotation DoF flags `{'x': bool, 'y': bool, 'z': bool}` |
| `m_dof_pose_enabled` | `bool` | Whether pose parameters can vary |
| `m_dof_pose_indices` | `List[int] \| None` | Joint indices for pose variation (None = all) |
| `m_dof_shape_enabled` | `bool` | Whether shape coefficients can vary |

### Member Variable Details

#### `m_model: Optional[smplx.SMPLX]`
The SMPL-X model instance that will be optimized. Must be provided through the factory method `from_smplx_model()`.

```python
# Model must be created by user
model = smplx.create(
    model_path='path/to/models',
    model_type='smplx',
    gender='neutral',
    batch_size=1
)
builder = SmplObjectiveBuilder.from_smplx_model(model)
```

#### `m_dof_translation: dict`
Dictionary controlling which translation axes can vary during optimization.

```python
# Default: all disabled
{'x': False, 'y': False, 'z': False}

# Enable all axes
builder.var_translation(True, True, True)
```

#### `m_dof_rotation: dict`
Dictionary controlling which rotation axes can vary during optimization.

```python
# Enable only Y-axis rotation (common for character turning)
builder.var_rotation(False, True, False)
```

#### `m_dof_pose_enabled: bool`
Master flag for pose parameter variation.

#### `m_dof_pose_indices: Optional[List[int]]`
Optional list of joint indices to restrict pose optimization. If `None`, all joints can vary.

```python
# Enable pose for specific joints only
upper_body_joints = [1, 2, 3, 4, 5]  # Example joint indices
builder.var_pose(True, indices=upper_body_joints)
```

#### `m_dof_shape_enabled: bool`
Flag controlling whether body shape parameters (betas) can vary.

## Properties

All properties provide read-only access to internal state and return copies to prevent external modification.

### `model: Optional[smplx.SMPLX]`
Get the SMPL-X model instance.

```python
current_model = builder.model
if current_model is not None:
    output = current_model()
```

### `dof_translation: dict`
Get translation degrees of freedom flags.

```python
translation_dof = builder.dof_translation
print(f"X translation enabled: {translation_dof['x']}")
```

### `dof_rotation: dict`
Get rotation degrees of freedom flags.

### `dof_pose_enabled: bool`
Get pose degrees of freedom enabled state.

### `dof_pose_indices: Optional[List[int]]`
Get pose degrees of freedom joint indices.

### `dof_shape_enabled: bool`
Get shape degrees of freedom enabled state.

## Class Methods

### `from_smplx_model(model: smplx.SMPLX) -> SmplObjectiveBuilder`

Factory method to create an instance from an existing SMPL-X model.

#### Parameters
- `model` (`smplx.SMPLX`): Existing SMPL-X model created by the user

#### Returns
- `SmplObjectiveBuilder`: New instance initialized with the given model

#### Example Usage

```python
import smplx
from hmodel_gen.smpl_objective_builder import SmplObjectiveBuilder

# User creates the SMPL-X model
model = smplx.create(
    model_path='path/to/smplx/models',
    model_type='smplx',
    gender='neutral',
    batch_size=1,
    use_pca=False,  # Disable PCA for full DoF
    flat_hand_mean=True  # Use flat hand pose as default
)

# Create objective builder
builder = SmplObjectiveBuilder.from_smplx_model(model)
```

!!! note "Model Requirements"
    The SMPL-X model must be created by the user. Common parameters:
    
    - `model_path`: Path to SMPL-X model files
    - `model_type`: Usually `'smplx'`
    - `gender`: `'neutral'`, `'male'`, or `'female'`
    - `batch_size`: Number of models in batch (typically 1)
    - `use_pca`: Set to `False` for full pose control

## Degree-of-Freedom Control Methods

These methods configure which parameters can vary during optimization and automatically set the `requires_grad` flag on corresponding model parameters.

### `var_translation(enable_x: bool, enable_y: bool, enable_z: bool) -> None`

Control whether translation parameters can vary.

#### Parameters
- `enable_x` (`bool`): Enable variation in x-axis translation
- `enable_y` (`bool`): Enable variation in y-axis translation
- `enable_z` (`bool`): Enable variation in z-axis translation

#### Example Usage

```python
# Enable translation in all axes
builder.var_translation(True, True, True)

# Enable only horizontal movement (X and Z)
builder.var_translation(True, False, True)

# Disable all translation
builder.var_translation(False, False, False)
```

### `var_rotation(enable_x: bool, enable_y: bool, enable_z: bool) -> None`

Control whether global rotation can vary about each axis.

#### Parameters
- `enable_x` (`bool`): Enable rotation about x-axis
- `enable_y` (`bool`): Enable rotation about y-axis
- `enable_z` (`bool`): Enable rotation about z-axis

#### Example Usage

```python
# Enable rotation around all axes
builder.var_rotation(True, True, True)

# Enable only turning (Y-axis rotation)
builder.var_rotation(False, True, False)

# Lock orientation completely
builder.var_rotation(False, False, False)
```

### `var_pose(enable: bool, indices: Optional[List[int]] = None) -> None`

Control whether pose parameters can vary.

#### Parameters
- `enable` (`bool`): Enable/disable pose variation
- `indices` (`List[int]`, optional): Joint indices to apply enable/disable to. None for all joints

#### Example Usage

```python
# Enable pose variation for all joints
builder.var_pose(True)

# Enable pose variation for specific joints only
upper_body_indices = [1, 2, 3, 12, 13, 14, 15, 16, 17]  # Example indices
builder.var_pose(True, indices=upper_body_indices)

# Disable all pose variation
builder.var_pose(False)
```

!!! info "SMPL-X Pose Parameters"
    The method affects multiple pose parameter tensors:
    
    - `body_pose`: Main body joint rotations
    - `jaw_pose`: Jaw movement
    - `leye_pose`, `reye_pose`: Eye rotations
    - `left_hand_pose`, `right_hand_pose`: Hand articulation

### `var_shape(enable: bool) -> None`

Control whether shape coefficients can vary.

#### Parameters
- `enable` (`bool`): Enable/disable shape variation

#### Example Usage

```python
# Enable shape variation
builder.var_shape(True)

# Lock body shape
builder.var_shape(False)
```

## Objective Building Functions

These methods create PyTorch tensors representing optimization objectives. All returned tensors have gradient tracking enabled.

### `obj_move_keypoints_to(smpl_keypoint_indices: List[int], target_points: torch.Tensor, weights: Union[torch.Tensor, float] = 1.0) -> torch.Tensor`

Build objective to move SMPL keypoints to target 3D locations.

#### Parameters
- `smpl_keypoint_indices` (`List[int]`): Indices of SMPL keypoints to move
- `target_points` (`torch.Tensor`): Target 3D positions with shape `(N, 3)` or `(1, N, 3)`
- `weights` (`torch.Tensor | float`, optional): Weight for each keypoint or single weight. Default: `1.0`

#### Returns
- `torch.Tensor`: Loss tensor for keypoint alignment

#### Raises
- `ValueError`: If model is not initialized

#### Example Usage

```python
# Move head to specific position
head_target = torch.tensor([[0.0, 1.8, 0.0]], dtype=torch.float32)
head_loss = builder.obj_move_keypoints_to(
    smpl_keypoint_indices=[15],  # Head joint index
    target_points=head_target,
    weights=1.0
)

# Move multiple keypoints with different weights
hand_targets = torch.tensor([
    [0.5, 1.0, 0.0],   # Right hand
    [-0.5, 1.0, 0.0]   # Left hand
], dtype=torch.float32)

hand_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
hand_loss = builder.obj_move_keypoints_to(
    smpl_keypoint_indices=[20, 21],  # Hand joint indices
    target_points=hand_targets,
    weights=hand_weights
)
```

!!! tip "SMPL-X Keypoint Indices"
    Common keypoint indices in SMPL-X:
    
    - 0: Pelvis
    - 15: Head
    - 20: Right hand
    - 21: Left hand
    - 7: Right foot
    - 10: Left foot
    
    Refer to SMPL-X documentation for complete joint mapping.

### `obj_l2_reg_translation(weights: Union[torch.Tensor, float]) -> torch.Tensor`

Build L2 regularization for translation vector.

#### Parameters
- `weights` (`torch.Tensor | float`): Shape `(3,)` tensor for x,y,z or single float for all dimensions

#### Returns
- `torch.Tensor`: L2 regularization loss for translation

#### Raises
- `ValueError`: If model is not initialized or has no translation parameter

#### Example Usage

```python
# Uniform regularization
translation_reg = builder.obj_l2_reg_translation(0.01)

# Per-axis regularization
axis_weights = torch.tensor([0.01, 0.1, 0.01], dtype=torch.float32)  # Stronger Y regularization
translation_reg = builder.obj_l2_reg_translation(axis_weights)
```

### `obj_l2_reg_rotation(weights: Union[torch.Tensor, float]) -> torch.Tensor`

Build L2 regularization for rotation vector.

#### Parameters
- `weights` (`torch.Tensor | float`): Shape `(3,)` tensor for rot_x,rot_y,rot_z or single float

#### Returns
- `torch.Tensor`: L2 regularization loss for rotation

#### Example Usage

```python
# Uniform rotation regularization
rotation_reg = builder.obj_l2_reg_rotation(0.05)

# Prevent excessive turning
turn_weights = torch.tensor([0.1, 0.01, 0.1], dtype=torch.float32)  # Allow Y rotation
rotation_reg = builder.obj_l2_reg_rotation(turn_weights)
```

### `obj_l2_reg_pose(weights: Union[torch.Tensor, float]) -> torch.Tensor`

Build L2 regularization for pose vector.

#### Parameters
- `weights` (`torch.Tensor | float`): Shape `(N,)` tensor for each joint or single float for all

#### Returns
- `torch.Tensor`: L2 regularization loss for pose parameters

#### Raises
- `ValueError`: If model is not initialized or weight shape doesn't match pose parameters

#### Example Usage

```python
# Uniform pose regularization
pose_reg = builder.obj_l2_reg_pose(0.01)

# Joint-specific regularization (if using pose indices)
builder.var_pose(True, indices=[1, 2, 3])  # 3 joints enabled
joint_weights = torch.tensor([0.1, 0.05, 0.1], dtype=torch.float32)
pose_reg = builder.obj_l2_reg_pose(joint_weights)
```

!!! warning "Weight Shape Validation"
    When using tensor weights with `dof_pose_indices`:
    
    - Weight tensor shape must match the number of enabled joints
    - Validation occurs at runtime and raises `ValueError` on mismatch

## Comprehensive Usage Examples

### Basic Pose Optimization

```python
import torch
import smplx
from hmodel_gen.smpl_objective_builder import SmplObjectiveBuilder

# Setup
model = smplx.create(model_path='models/', model_type='smplx', gender='neutral')
builder = SmplObjectiveBuilder.from_smplx_model(model)

# Configure DoF
builder.var_translation(True, True, True)
builder.var_rotation(False, True, False)  # Only Y rotation
builder.var_pose(True)

# Define target
target_position = torch.tensor([[0.0, 1.8, 0.0]], dtype=torch.float32)

# Setup optimizer
optimizer = torch.optim.Adam([
    model.transl,
    model.global_orient,
    model.body_pose
], lr=0.01)

# Optimization loop
for iteration in range(100):
    optimizer.zero_grad()
    
    # Primary objective: head position
    head_loss = builder.obj_move_keypoints_to([15], target_position, 1.0)
    
    # Regularization objectives
    pose_reg = builder.obj_l2_reg_pose(0.01)
    trans_reg = builder.obj_l2_reg_translation(0.1)
    
    # Combine objectives
    total_loss = head_loss + pose_reg + trans_reg
    
    total_loss.backward()
    optimizer.step()
    
    if iteration % 20 == 0:
        print(f"Iteration {iteration}: Loss = {total_loss.item():.6f}")

# Get final result
final_output = model()
print(f"Final head position: {final_output.joints[0, 15, :]}")
```

### Multi-Target Optimization

```python
# Multiple keypoint targets
targets = {
    'head': ([15], torch.tensor([[0.0, 1.8, 0.0]])),
    'hands': ([20, 21], torch.tensor([[0.3, 1.2, 0.2], [-0.3, 1.2, 0.2]])),
    'feet': ([7, 10], torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]))
}

# Configure builder
builder.var_translation(True, True, True)
builder.var_pose(True)

# Create optimizer
optimizer = torch.optim.Adam([model.transl, model.body_pose], lr=0.01)

for iteration in range(200):
    optimizer.zero_grad()
    
    total_loss = torch.tensor(0.0, dtype=torch.float32)
    
    # Add all keypoint objectives
    for name, (indices, positions) in targets.items():
        loss = builder.obj_move_keypoints_to(indices, positions, 1.0)
        total_loss += loss
    
    # Add regularization
    total_loss += builder.obj_l2_reg_pose(0.01)
    total_loss += builder.obj_l2_reg_translation(0.1)
    
    total_loss.backward()
    optimizer.step()
    
    if iteration % 50 == 0:
        print(f"Iteration {iteration}: Total loss = {total_loss.item():.6f}")
```

### Progressive Optimization

```python
def progressive_optimization(builder: SmplObjectiveBuilder, targets: dict):
    """Optimize in stages for better convergence."""
    
    # Stage 1: Global positioning
    print("Stage 1: Global positioning")
    builder.var_translation(True, True, True)
    builder.var_rotation(True, True, True)
    builder.var_pose(False)
    
    global_optimizer = torch.optim.Adam([
        builder.model.transl,
        builder.model.global_orient
    ], lr=0.1)
    
    for i in range(50):
        global_optimizer.zero_grad()
        loss = builder.obj_move_keypoints_to(
            targets['indices'], targets['positions'], 1.0
        )
        loss.backward()
        global_optimizer.step()
        
        if i % 10 == 0:
            print(f"  Global iter {i}: {loss.item():.6f}")
    
    # Stage 2: Detailed pose
    print("Stage 2: Detailed pose")
    builder.var_pose(True)
    
    pose_optimizer = torch.optim.Adam([builder.model.body_pose], lr=0.01)
    
    for i in range(100):
        pose_optimizer.zero_grad()
        
        keypoint_loss = builder.obj_move_keypoints_to(
            targets['indices'], targets['positions'], 1.0
        )
        reg_loss = builder.obj_l2_reg_pose(0.01)
        
        total_loss = keypoint_loss + reg_loss
        total_loss.backward()
        pose_optimizer.step()
        
        if i % 20 == 0:
            print(f"  Pose iter {i}: {total_loss.item():.6f}")
    
    return builder.model()

# Usage
optimization_targets = {
    'indices': [15, 20, 21],
    'positions': torch.tensor([
        [0.0, 1.8, 0.0],    # Head
        [0.3, 1.2, 0.2],    # Right hand
        [-0.3, 1.2, 0.2]    # Left hand
    ])
}

result = progressive_optimization(builder, optimization_targets)
```

### Shape and Pose Co-optimization

```python
def optimize_shape_and_pose(builder: SmplObjectiveBuilder, 
                           keypoint_targets: torch.Tensor,
                           keypoint_indices: List[int]):
    """Jointly optimize shape and pose parameters."""
    
    # Enable all relevant DoF
    builder.var_translation(True, True, True)
    builder.var_pose(True)
    builder.var_shape(True)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': [builder.model.transl], 'lr': 0.01},
        {'params': [builder.model.body_pose], 'lr': 0.01},
        {'params': [builder.model.betas], 'lr': 0.001}  # Slower for shape
    ]
    
    optimizer = torch.optim.Adam(param_groups)
    
    for iteration in range(300):
        optimizer.zero_grad()
        
        # Primary objective
        keypoint_loss = builder.obj_move_keypoints_to(
            keypoint_indices, keypoint_targets, 1.0
        )
        
        # Regularization with different strengths
        pose_reg = builder.obj_l2_reg_pose(0.01)
        trans_reg = builder.obj_l2_reg_translation(0.1)
        
        # Strong shape regularization to prevent unrealistic bodies
        shape_reg = torch.sum(builder.model.betas ** 2) * 0.5
        
        total_loss = keypoint_loss + pose_reg + trans_reg + shape_reg
        
        total_loss.backward()
        optimizer.step()
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Loss = {total_loss.item():.6f}")
            print(f"  Keypoint: {keypoint_loss.item():.6f}")
            print(f"  Shape reg: {shape_reg.item():.6f}")
    
    return {
        'model_output': builder.model(),
        'final_shape': builder.model.betas.detach().clone(),
        'final_pose': builder.model.body_pose.detach().clone()
    }
```

### Custom Objective Integration

```python
class ExtendedObjectiveBuilder(SmplObjectiveBuilder):
    """Extended builder with custom objectives."""
    
    def obj_bone_length_constraint(self, bone_pairs: List[tuple], 
                                 target_lengths: torch.Tensor,
                                 weights: torch.Tensor) -> torch.Tensor:
        """Custom objective to maintain specific bone lengths."""
        if self.m_model is None:
            raise ValueError("Model not initialized")
        
        model_output = self.m_model()
        joints = model_output.joints[0]  # Shape: (n_joints, 3)
        
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        
        for i, (joint1_idx, joint2_idx) in enumerate(bone_pairs):
            # Calculate current bone length
            joint1_pos = joints[joint1_idx]
            joint2_pos = joints[joint2_idx]
            current_length = torch.norm(joint1_pos - joint2_pos)
            
            # Compare to target
            target_length = target_lengths[i]
            weight = weights[i]
            
            loss = weight * (current_length - target_length) ** 2
            total_loss += loss
        
        return total_loss
    
    def obj_symmetry_constraint(self, left_indices: List[int], 
                              right_indices: List[int],
                              weight: float = 1.0) -> torch.Tensor:
        """Enforce symmetry between left and right body parts."""
        if self.m_model is None:
            raise ValueError("Model not initialized")
        
        model_output = self.m_model()
        joints = model_output.joints[0]
        
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        
        for left_idx, right_idx in zip(left_indices, right_indices):
            left_pos = joints[left_idx]
            right_pos = joints[right_idx]
            
            # Mirror right position across YZ plane (negate X)
            mirrored_right = torch.tensor([-right_pos[0], right_pos[1], right_pos[2]])
            
            # Compute symmetry loss
            symmetry_loss = torch.sum((left_pos - mirrored_right) ** 2)
            total_loss += weight * symmetry_loss
        
        return total_loss

# Usage of extended builder
extended_builder = ExtendedObjectiveBuilder.from_smplx_model(model)
extended_builder.var_pose(True)

# Define bone length constraints
bone_pairs = [(1, 4), (2, 5)]  # Example joint pairs
target_lengths = torch.tensor([0.3, 0.4], dtype=torch.float32)
bone_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

# Define symmetry constraints
left_joints = [20, 7]   # Left hand, left foot
right_joints = [21, 10] # Right hand, right foot

# Optimization with custom objectives
optimizer = torch.optim.Adam([model.body_pose], lr=0.01)

for iteration in range(100):
    optimizer.zero_grad()
    
    # Standard objectives
    keypoint_loss = extended_builder.obj_move_keypoints_to([15], target_head, 1.0)
    pose_reg = extended_builder.obj_l2_reg_pose(0.01)
    
    # Custom objectives
    bone_loss = extended_builder.obj_bone_length_constraint(
        bone_pairs, target_lengths, bone_weights
    )
    symmetry_loss = extended_builder.obj_symmetry_constraint(
        left_joints, right_joints, 0.5
    )
    
    total_loss = keypoint_loss + pose_reg + bone_loss + symmetry_loss
    
    total_loss.backward()
    optimizer.step()
```

## Integration Patterns

### With Motion Capture Data

```python
from hmodel_gen.bvh_parser import BVHData

def fit_mocap_sequence(bvh_data: BVHData, model: smplx.SMPLX) -> List[dict]:
    """Fit SMPL-X to BVH motion capture data."""
    builder = SmplObjectiveBuilder.from_smplx_model(model)
    builder.var_translation(True, True, True)
    builder.var_pose(True)
    
    results = []
    
    # Map BVH joints to SMPL keypoints (simplified mapping)
    joint_mapping = {
        'Hips': 0,
        'Head': 15,
        'LeftHand': 21,
        'RightHand': 20
    }
    
    for frame_idx in range(min(100, bvh_data.n_frames)):  # Limit for example
        # Reset model parameters
        model.transl.data.zero_()
        model.body_pose.data.zero_()
        
        # Get BVH pose
        bvh_pose = bvh_data.get_pose(frame_idx)
        bvh_positions = bvh_pose.get_joint_positions("world")
        
        # Create target positions for SMPL
        target_indices = []
        target_positions = []
        
        for bvh_joint, smpl_idx in joint_mapping.items():
            if bvh_joint in bvh_positions:
                target_indices.append(smpl_idx)
                target_positions.append(bvh_positions[bvh_joint])
        
        target_tensor = torch.tensor(target_positions, dtype=torch.float32)
        
        # Optimize for this frame
        optimizer = torch.optim.LBFGS([model.transl, model.body_pose], lr=0.1)
        
        def closure():
            optimizer.zero_grad()
            loss = builder.obj_move_keypoints_to(target_indices, target_tensor, 1.0)
            loss += builder.obj_l2_reg_pose(0.01)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Store result
        model_output = model()
        results.append({
            'frame': frame_idx,
            'vertices': model_output.vertices.detach().clone(),
            'joints': model_output.joints.detach().clone(),
            'translation': model.transl.detach().clone(),
            'pose': model.body_pose.detach().clone()
        })
    
    return results
```

### With Real-time Control

```python
class RealTimeController:
    """Real-time SMPL-X character controller."""
    
    def __init__(self, model: smplx.SMPLX):
        self.model = model
        self.builder = SmplObjectiveBuilder.from_smplx_model(model)
        self.builder.var_pose(True)
        
        # Pre-configure optimizer for fast updates
        self.optimizer = torch.optim.Adam([model.body_pose], lr=0.1)
        
        # Target cache
        self.current_targets = {}
    
    def set_target(self, joint_name: str, position: torch.Tensor, weight: float = 1.0):
        """Set target position for a joint."""
        # Map joint names to indices (simplified)
        joint_map = {'head': 15, 'left_hand': 21, 'right_hand': 20}
        
        if joint_name in joint_map:
            self.current_targets[joint_name] = {
                'index': joint_map[joint_name],
                'position': position,
                'weight': weight
            }
    
    def update(self, max_iterations: int = 10) -> dict:
        """Update pose to match current targets."""
        if not self.current_targets:
            return {'success': False, 'message': 'No targets set'}
        
        for iteration in range(max_iterations):
            self.optimizer.zero_grad()
            
            total_loss = torch.tensor(0.0, dtype=torch.float32)
            
            # Process all current targets
            for target_data in self.current_targets.values():
                loss = self.builder.obj_move_keypoints_to(
                    [target_data['index']],
                    target_data['position'].unsqueeze(0),
                    target_data['weight']
                )
                total_loss += loss
            
            # Light regularization for stability
            total_loss += self.builder.obj_l2_reg_pose(0.001)
            
            total_loss.backward()
            self.optimizer.step()
        
        return {
            'success': True,
            'final_loss': total_loss.item(),
            'model_output': self.model()
        }
    
    def clear_targets(self):
        """Clear all current targets."""
        self.current_targets.clear()

# Usage
controller = RealTimeController(model)

# Set targets
controller.set_target('head', torch.tensor([0.0, 1.8, 0.0]), weight=1.0)
controller.set_target('right_hand', torch.tensor([0.5, 1.2, 0.0]), weight=0.8)

# Update pose
result = controller.update(max_iterations=5)
if result['success']:
    final_pose = result['model_output']
```

## Performance Considerations

### Memory Management

```python
def optimize_with_memory_management(builder: SmplObjectiveBuilder, 
                                  targets: dict, 
                                  max_iterations: int = 100):
    """Optimization with explicit memory management."""
    
    # Enable gradient checkpointing for large models
    if hasattr(builder.model, 'use_face_contour'):
        builder.model.use_face_contour = False  # Reduce memory
    
    # Use memory-efficient optimizer
    optimizer = torch.optim.AdamW([
        builder.model.transl,
        builder.model.body_pose
    ], lr=0.01, weight_decay=1e-5)
    
    for iteration in range(max_iterations):
        # Clear gradients
        optimizer.zero_grad()
        
        # Compute loss with gradient accumulation if needed
        loss = builder.obj_move_keypoints_to(
            targets['indices'], targets['positions'], 1.0
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([
            builder.model.transl,
            builder.model.body_pose
        ], max_norm=1.0)
        
        optimizer.step()
        
        # Manual garbage collection for long optimizations
        if iteration % 50 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if iteration % 20 == 0:
            print(f"Iteration {iteration}: Loss = {loss.item():.6f}")
    
    return builder.model()
```

### Batch Processing

```python
def batch_optimization(models: List[smplx.SMPLX], 
                      target_sequences: List[torch.Tensor]) -> List[dict]:
    """Optimize multiple models simultaneously."""
    
    results = []
    
    for model, targets in zip(models, target_sequences):
        builder = SmplObjectiveBuilder.from_smplx_model(model)
        builder.var_translation(True, True, True)
        builder.var_pose(True)
        
        # Batch optimization parameters
        optimizer = torch.optim.Adam([
            model.transl,
            model.body_pose
        ], lr=0.01)
        
        # Quick optimization for batch processing
        for iteration in range(50):  # Fewer iterations per model
            optimizer.zero_grad()
            
            loss = builder.obj_move_keypoints_to(
                [15], targets, 1.0  # Just head for speed
            )
            loss += builder.obj_l2_reg_pose(0.01)
            
            loss.backward()
            optimizer.step()
        
        # Store results
        model_output = model()
        results.append({
            'vertices': model_output.vertices.detach(),
            'joints': model_output.joints.detach(),
            'final_loss': loss.item()
        })
    
    return results
```

## Best Practices

!!! tip "Optimization Strategy"
    - Use progressive optimization: global positioning → detailed pose → shape
    - Start with fewer iterations and increase as needed
    - Combine multiple objectives with appropriate weights
    - Use L-BFGS for final refinement after Adam initialization

!!! warning "Common Pitfalls"
    - Don't optimize parameters without enabling corresponding DoF
    - Avoid very large regularization weights that dominate objectives
    - Check for gradient flow issues if optimization stalls
    - Ensure target positions are in reasonable coordinate ranges

!!! info "Performance Tips"
    - Use `requires_grad=False` for parameters you don't need to optimize
    - Consider gradient checkpointing for memory-constrained environments
    - Batch operations when possible for better GPU utilization
    - Profile optimization loops to identify bottlenecks

## Error Handling and Debugging

### Common Errors

```python
def debug_optimization_setup(builder: SmplObjectiveBuilder):
    """Debug common optimization setup issues."""
    
    if builder.model is None:
        print("ERROR: Model not initialized. Use from_smplx_model() first.")
        return False
    
    # Check if any DoF are enabled
    has_dof = (
        any(builder.dof_translation.values()) or
        any(builder.dof_rotation.values()) or
        builder.dof_pose_enabled or
        builder.dof_shape_enabled
    )
    
    if not has_dof:
        print("WARNING: No degrees of freedom enabled. Nothing will optimize.")
    
    # Check gradient requirements
    param_grads = []
    if hasattr(builder.model, 'transl'):
        param_grads.append(f"transl: {builder.model.transl.requires_grad}")
    if hasattr(builder.model, 'global_orient'):
        param_grads.append(f"global_orient: {builder.model.global_orient.requires_grad}")
    if hasattr(builder.model, 'body_pose'):
        param_grads.append(f"body_pose: {builder.model.body_pose.requires_grad}")
    
    print("Parameter gradient requirements:")
    for param_grad in param_grads:
        print(f"  {param_grad}")
    
    return True

# Usage before optimization
debug_optimization_setup(builder)
```

### Validation Functions

```python
def validate_targets(indices: List[int], positions: torch.Tensor):
    """Validate target positions for optimization."""
    
    # Check indices range
    if any(idx < 0 for idx in indices):
        raise ValueError("Joint indices must be non-negative")
    
    # Check position tensor shape
    if positions.dim() not in [2, 3]:
        raise ValueError("Target positions must be 2D or 3D tensor")
    
    if positions.shape[-1] != 3:
        raise ValueError("Target positions must have 3 coordinates")
    
    # Check for reasonable position values
    pos_magnitude = torch.norm(positions, dim=-1)
    if torch.any(pos_magnitude > 5.0):  # 5 meters seems unreasonable
        print("WARNING: Some target positions are very far from origin")
    
    print(f"Validation passed: {len(indices)} targets, shape {positions.shape}")

# Usage
validate_targets([15, 20, 21], target_positions)
```