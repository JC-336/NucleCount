# Nuclei Detection and Counting Tool

A computer vision-based tool for automated detection and counting of cell nuclei in microscopy images. This tool is designed to identify both circular and oval-shaped nuclei, handle overlapping structures, and distinguish between complete nuclei and those touching image borders.

---

## 🔧 Features

- Automated detection of cell nuclei in microscopy images
- **Shape classification**: Identifies both circular and oval-shaped nuclei
- **Overlap handling**: Merges overlapping small circles into ovals
- **Border detection**: Differentiates between complete nuclei and those touching image borders
- **Blue content verification**: Validates potential nuclei based on staining intensity
- **Scale bar removal**: Automatically excludes scale bars from analysis
- **Comprehensive visualization**: Color-coded results with detailed legend
- **Data export**: Results compiled into CSV for further analysis

---

## 📦 Installation Requirements
## 🐍 Python Compatibility

This project is compatible with **Python 3.6 and above**.

### ✅ Requirements and Dependencies

Before running the project, ensure you have Python 3 installed. You can check your version with:

```bash
python --version
```

```bash
pip install opencv-python numpy pandas matplotlib scikit-image
```
