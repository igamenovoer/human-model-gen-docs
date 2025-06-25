# Blender Tools - Human-Friendly Remote Control APIs

The `blender_tools` directory provides Python APIs for human-friendly remote control of Blender via the MCP (Model Context Protocol) server. These tools enable programmatic manipulation of Blender scenes, asset management, and rendering operations through clean, Pythonic interfaces.

## Overview

The Blender MCP Client provides three main classes for different aspects of Blender control:

- **`BlenderMCPClient`** - Low-level MCP communication with automatic environment detection
- **`BlenderAssetManager`** - Asset library management, browsing, and import functionality
- **`BlenderSceneManager`** - 3D scene manipulation, object creation, and rendering

## Prerequisites

1. **Blender MCP Server**: Must be running on port 9876
2. **Asset Libraries**: Configured in Blender preferences
3. **Python Environment**: Standard library with NumPy support
4. **Network Access**: TCP socket communication to Blender

## Quick Start

```python
from blender_tools.blender_mcp_client import quick_scene_setup

# One-liner setup for all functionality
client, assets, scene = quick_scene_setup()

# Clear scene and add objects
scene.clear_scene()
cube = scene.add_cube(location=(0, 0, 0), name="MyCube")
sphere = scene.add_sphere(location=(3, 0, 0), name="MySphere")

# Position camera and take screenshot
scene.set_camera_location(location=(7, -7, 5), target=(0, 0, 0))
client.take_screenshot("/tmp/my_scene.png")
```

## Core Classes

### BlenderMCPClient

Low-level client for MCP communication with automatic environment detection.

**Key Features:**
- Direct TCP socket communication (no subprocess dependencies)
- Auto-detects Docker vs local environment
- Robust error handling and timeout management
- Comprehensive response parsing

**Key Methods:**

```python
client = BlenderMCPClient()

# Basic operations
scene_info = client.get_scene_info()
result = client.execute_python("import bpy; print(bpy.app.version)")
screenshot = client.take_screenshot("/tmp/viewport.png", max_size=1920)
connection_ok = client.test_connection()

# Raw command execution
response = client.execute_command("get_object_info", {"name": "Cube"})
```

**Environment Detection:**
- **Docker**: Automatically uses `host.docker.internal` for MCP connections
- **Local**: Uses `localhost` for MCP connections
- **Custom**: Override with explicit host parameter

### BlenderAssetManager

High-level asset library management for browsing and importing Blender assets.

**Key Methods:**

```python
assets = BlenderAssetManager(client)

# List all configured asset libraries
libraries = assets.list_asset_libraries()
# Returns: [{"name": "blender-assets", "path": "/path/to/assets"}, ...]

# Browse collections in a specific library
collections = assets.list_library_collections("blender-assets")
# Returns: [{"file": "railings.blend", "collections": ["Wooden Fence", ...]}, ...]

# List catalogs (directories and .blend files) in library
catalogs = assets.list_library_catalogs("KitBash3D")
# Returns: {"directories": [...], "blend_files": [...], "summary": {...}}

# Import a specific collection
success = assets.import_collection("blender-assets", "railings.blend", "Wooden Fence")
```

**Asset Library Examples:**
- `"blender-assets"` - E:\wps-drive\blender-assets
- `"User Library"` - C:\Users\username\Documents\Blender\Assets
- `"Poly Haven"` - G:\asset3d\Poly Haven
- `"KitBash3D"` - G:\BaiduNetdiskDownload\blender-scenes\KitBash3D

### BlenderSceneManager

High-level 3D scene manipulation with full NumPy integration.

**Scene Information:**

```python
scene = BlenderSceneManager(client)

# Get scene summary
summary = scene.get_scene_summary()

# List objects with optional filtering
all_objects = scene.list_objects()
meshes_only = scene.list_objects(object_type="MESH")
cameras_only = scene.list_objects(object_type="CAMERA")

# Each object returns:
# {
#   "name": "ObjectName",
#   "type": "MESH",
#   "location": numpy.array([x, y, z]),
#   "rotation": numpy.array([rx, ry, rz]),
#   "scale": numpy.array([sx, sy, sz]),
#   "visible": True
# }
```

**Scene Manipulation:**

```python
import numpy as np

# Clear scene (with options)
scene.clear_scene(keep_camera=True, keep_light=True)

# Add primitive objects with NumPy arrays
cube_name = scene.add_cube(
    location=np.array([0, 0, 0]), 
    size=2.0, 
    name="MyCube"
)

sphere_name = scene.add_sphere(
    location=np.array([3, 0, 0]), 
    radius=1.0, 
    name="MySphere"
)

cylinder_name = scene.add_cylinder(
    location=np.array([-3, 0, 0]), 
    radius=0.8, 
    depth=3.0, 
    name="MyCylinder"
)

# Generic primitive creation
object_name = scene.add_primitive(
    primitive_type="uv_sphere",
    location=np.array([0, 0, 0]),
    rotation=np.array([0, 0, 0]),  # radians
    scale=np.array([1, 1, 1]),
    name="MyObject"
)
```

**Object Operations:**

```python
# Move objects
success = scene.move_object("MyCube", np.array([5, 5, 0]))

# Delete objects
success = scene.delete_object("OldObject")

# Camera control
scene.set_camera_location(
    location=np.array([7, -7, 5]), 
    target=np.array([0, 0, 0])
)
```

**Rendering:**

```python
# Take viewport screenshot
screenshot_info = client.take_screenshot(
    filepath="/tmp/viewport.png",
    max_size=1920,
    format="png"
)

# Render scene to image
success = scene.render_image(
    filepath="/tmp/render.png",
    resolution=np.array([1920, 1080])
)
```

## Advanced Usage

### Error Handling

All operations can raise `BlenderMCPError` exceptions:

```python
from blender_tools.blender_mcp_client import BlenderMCPError

try:
    client = connect_to_blender()
    scene = BlenderSceneManager(client)
    scene.add_cube(location=(0, 0, 0))
except BlenderMCPError as e:
    print(f"Blender operation failed: {e}")
```

### Custom Python Execution

For operations not covered by the high-level APIs:

```python
# Execute custom Python code in Blender
custom_code = """
import bpy
import bmesh

# Create a custom mesh
bpy.ops.mesh.primitive_monkey_add()
obj = bpy.context.active_object
obj.name = "CustomMonkey"

# Apply modifiers
bpy.ops.object.modifier_add(type='SUBSURF')
"""

result = client.execute_python(custom_code)
print(result)
```

### Batch Operations

```python
# Import multiple assets and position them
assets_to_import = [
    ("railings.blend", "Wooden Fence"),
    ("railings.blend", "Cast Iron Fence"),
    ("railings.blend", "Metal Gate")
]

imported_objects = []
for file_path, collection_name in assets_to_import:
    success = assets.import_collection("blender-assets", file_path, collection_name)
    if success:
        imported_objects.append(collection_name)

# Position objects in a line
for i, obj_name in enumerate(imported_objects):
    position = np.array([i * 3.0, 0, 0])  # 3 units apart
    scene.move_object(obj_name, position)
```

### NumPy Integration

The tools provide full NumPy integration for 3D operations:

```python
import numpy as np

# Create objects with precise positioning
positions = np.array([
    [0, 0, 0],
    [2, 0, 0], 
    [4, 0, 0],
    [6, 0, 0]
])

for i, pos in enumerate(positions):
    cube_name = scene.add_cube(
        location=pos,
        size=1.0,
        name=f"Cube_{i}"
    )

# Move objects in patterns
objects = scene.list_objects(object_type="MESH")
for i, obj in enumerate(objects):
    # Create circular arrangement
    angle = i * 2 * np.pi / len(objects)
    radius = 5.0
    new_pos = np.array([
        radius * np.cos(angle),
        radius * np.sin(angle),
        0
    ])
    scene.move_object(obj['name'], new_pos)
```

## Examples and Use Cases

### 1. Scene Setup and Asset Import

```python
from blender_tools.blender_mcp_client import quick_scene_setup
import numpy as np

# Setup
client, assets, scene = quick_scene_setup()

# Clear and setup scene
scene.clear_scene()

# Import assets
libraries = assets.list_asset_libraries()
for lib in libraries:
    if lib['name'] == 'blender-assets':
        collections = assets.list_library_collections(lib['name'])
        for item in collections[:3]:  # Import first 3
            assets.import_collection(lib['name'], item['file'], item['collections'][0])

# Position camera for overview
scene.set_camera_location(
    location=np.array([20, -20, 15]), 
    target=np.array([0, 0, 0])
)
```

### 2. Procedural Scene Generation

```python
# Create a grid of objects
grid_size = 5
spacing = 3.0

for x in range(grid_size):
    for y in range(grid_size):
        position = np.array([x * spacing, y * spacing, 0])
        
        # Alternate between cubes and spheres
        if (x + y) % 2 == 0:
            scene.add_cube(location=position, size=1.0, name=f"Cube_{x}_{y}")
        else:
            scene.add_sphere(location=position, radius=0.5, name=f"Sphere_{x}_{y}")

# Take overview screenshot
client.take_screenshot("/tmp/procedural_scene.png", max_size=1920)
```

### 3. Asset Library Browser

```python
# Browse all available assets
def browse_all_assets():
    libraries = assets.list_asset_libraries()
    
    for lib in libraries:
        print(f"\nLibrary: {lib['name']}")
        print(f"Path: {lib['path']}")
        
        # Get catalogs overview
        catalogs = assets.list_library_catalogs(lib['name'])
        print(f"Directories: {catalogs['summary']['directory_count']}")
        print(f"Blend files: {catalogs['summary']['blend_count']}")
        
        # List some collections
        collections = assets.list_library_collections(lib['name'])
        print("Sample collections:")
        for item in collections[:3]:
            print(f"  {item['file']}: {len(item['collections'])} collections")

browse_all_assets()
```

### 4. Animation Setup

```python
# Create objects for animation
objects_data = [
    {"name": "Bouncer", "pos": [0, 0, 0], "type": "sphere"},
    {"name": "Rotator", "pos": [5, 0, 0], "type": "cube"},
    {"name": "Slider", "pos": [10, 0, 0], "type": "cylinder"}
]

for obj_data in objects_data:
    pos = np.array(obj_data["pos"])
    
    if obj_data["type"] == "sphere":
        scene.add_sphere(location=pos, name=obj_data["name"])
    elif obj_data["type"] == "cube":
        scene.add_cube(location=pos, name=obj_data["name"])
    elif obj_data["type"] == "cylinder":
        scene.add_cylinder(location=pos, name=obj_data["name"])

# Position camera for animation
scene.set_camera_location(
    location=np.array([15, -10, 8]),
    target=np.array([5, 0, 0])
)
```

## Integration with HumanModelGeneration Project

The blender_tools integrate seamlessly with the existing project workflow:

```python
# Combined workflow example
from blender_tools.blender_mcp_client import quick_scene_setup
from blender_skel import RiggedMesh

# Setup Blender remote control
client, assets, scene = quick_scene_setup()

# Load project data
mesh = RiggedMesh.from_blender_exported_dir("output-uuid/")

# Setup Blender scene for visualization
scene.clear_scene()

# Import relevant assets for context
assets.import_collection("blender-assets", "railings.blend", "Simple Wooden Fence")

# Position camera for human model visualization
scene.set_camera_location(
    location=np.array([3, -3, 2]), 
    target=np.array([0, 0, 1])
)

# Take screenshot for documentation
client.take_screenshot("/tmp/human_model_context.png")
```

## API Reference

### BlenderMCPClient

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `host=None, port=9876, timeout=30.0` | `BlenderMCPClient` | Initialize client with auto-detection |
| `execute_command` | `command_type: str, params: dict` | `dict` | Execute raw MCP command |
| `execute_python` | `code: str` | `str` | Execute Python code in Blender |
| `get_scene_info` | - | `dict` | Get current scene information |
| `get_object_info` | `object_name: str` | `dict` | Get specific object information |
| `take_screenshot` | `filepath: str, max_size=1920, format="png"` | `dict` | Capture viewport screenshot |
| `test_connection` | - | `bool` | Test server connectivity |

### BlenderAssetManager

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `client: BlenderMCPClient` | `BlenderAssetManager` | Initialize with MCP client |
| `list_asset_libraries` | - | `List[dict]` | List all configured libraries |
| `list_library_collections` | `library_name: str` | `List[dict]` | List collections in library |
| `list_library_catalogs` | `library_name: str` | `dict` | List catalogs in library |
| `import_collection` | `library_name: str, file_path: str, collection_name: str` | `bool` | Import specific collection |

### BlenderSceneManager

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `client: BlenderMCPClient` | `BlenderSceneManager` | Initialize with MCP client |
| `list_objects` | `object_type=None` | `List[dict]` | List scene objects |
| `clear_scene` | `keep_camera=True, keep_light=True` | `bool` | Clear scene objects |
| `add_cube` | `location, size=2.0, name=None` | `str` | Add cube primitive |
| `add_sphere` | `location, radius=1.0, name=None` | `str` | Add sphere primitive |
| `add_cylinder` | `location, radius=1.0, depth=2.0, name=None` | `str` | Add cylinder primitive |
| `add_primitive` | `primitive_type, location, rotation, scale, name` | `str` | Add generic primitive |
| `delete_object` | `object_name: str` | `bool` | Delete object by name |
| `move_object` | `object_name: str, location` | `bool` | Move object to location |
| `set_camera_location` | `location, target=None` | `bool` | Position camera |
| `render_image` | `filepath: str, resolution=None` | `bool` | Render scene to image |

## Troubleshooting

### Connection Issues
- Ensure Blender MCP server is running on port 9876
- Check firewall settings for port access
- Verify Docker networking if running in container
- Use `client.test_connection()` to verify connectivity

### Import Failures
- Check asset library paths in Blender preferences
- Verify .blend file integrity and accessibility
- Ensure collection names exist in target files
- Check for file path encoding issues on Windows

### Rendering Issues
- Check output directory permissions
- Verify render engine settings in Blender
- Ensure sufficient disk space for output files
- Test with smaller resolution first

### Performance Optimization
- Use batch operations when possible
- Cache asset library information
- Minimize individual object operations
- Use appropriate timeout values for large operations

## Development Guidelines

### Adding New Functionality

1. **Low-level operations**: Add to `BlenderMCPClient`
2. **Asset operations**: Add to `BlenderAssetManager`  
3. **Scene operations**: Add to `BlenderSceneManager`
4. **Follow patterns**: Use consistent JSON parsing patterns
5. **Error handling**: Always use try/catch with meaningful errors

### API Design Principles

- **Human-friendly**: Clear method names, sensible defaults
- **Type hints**: Full typing support for IDE assistance
- **NumPy integration**: Support numpy arrays for 3D operations
- **Documentation**: Comprehensive docstrings with examples
- **Consistency**: Similar patterns across all methods
- **Robustness**: Graceful error handling and recovery

## See Also

- [Blender MCP CLI Guide](../ai-context/summary/blender-mcp-cli-guide.md)
- [Blender Python API Documentation](https://docs.blender.org/api/current/index.html)
- [HumanModelGeneration Main Documentation](../README.md)
- [Project Architecture Overview](../docs/architecture.md)