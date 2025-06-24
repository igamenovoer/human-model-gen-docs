# HumanModelGeneration Documentation

This repository hosts the documentation for the `hmodel_gen` module classes from the HumanModelGeneration project.

## 📚 Documentation Contents

The documentation covers all classes in the `hmodel_gen` module:

### BVH Parser Module
- **SingleJoint**: Individual joint representation with transformation matrices
- **SinglePose**: Complete pose data for a single frame  
- **BVHData**: Standard BVH data structure for motion capture files
- **BVHFlatData**: Memory-efficient storage for large animations

### SMPL Objective Builder Module
- **SmplObjectiveBuilder**: PyTorch-based optimization objectives for SMPL-X models

## 🌐 Live Documentation

The documentation is automatically built and deployed to GitHub Pages:

**[📖 View Documentation](https://igamenovoer.github.io/human-model-gen-docs)**

## 🔧 Local Development

To build and serve the documentation locally:

1. Install MkDocs Material:
   ```bash
   pip install mkdocs-material
   ```

2. Serve locally:
   ```bash
   mkdocs serve
   ```

3. Build static site:
   ```bash
   mkdocs build
   ```

## 🚀 Deployment

The documentation is automatically deployed using GitHub Actions when changes are pushed to the main branch. The workflow:

1. Installs Python and MkDocs Material
2. Builds the documentation using `mkdocs build`
3. Deploys to GitHub Pages

## 📝 Documentation Structure

```
.
├── index.md                          # Home page
├── bvh_parser/
│   ├── index.md                      # BVH Parser module overview
│   ├── single_joint.md               # SingleJoint class documentation
│   ├── single_pose.md                # SinglePose class documentation
│   ├── bvh_data.md                   # BVHData class documentation
│   └── bvh_flat_data.md              # BVHFlatData class documentation
├── smpl_objective_builder/
│   ├── index.md                      # SMPL module overview
│   └── smpl_objective_builder.md     # SmplObjectiveBuilder class documentation
└── mkdocs.yml                        # MkDocs configuration
```

## 🎨 Features

- **Material Design**: Clean, modern interface with dark/light theme toggle
- **Search**: Full-text search across all documentation
- **Navigation**: Organized tabs and sections for easy browsing
- **Code Highlighting**: Syntax highlighting for Python code examples
- **Mobile Responsive**: Works on all device sizes
- **Mermaid Diagrams**: Architecture diagrams and flowcharts

## 🤝 Contributing

This repository contains only documentation files. The source code is maintained in a separate private repository.

To suggest documentation improvements:
1. Open an issue describing the improvement
2. Or submit a pull request with your changes

## 📄 License

This documentation is part of the HumanModelGeneration project.
