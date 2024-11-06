# MeshPress

MeshPress is a 3D mesh compression library that offers various encoding techniques to reduce the storage size of 3D models. The project aims to provide efficient compression of 3D meshes while preserving the geometric integrity of the models. This is especially useful in applications such as graphics rendering, AR/VR, and game development where memory efficiency and loading times are critical.

## Features

- Multiple encoding techniques with varying levels of compression efficiency.
- Ability to choose different packing strategies to optimize for space.
- Statistics on compression performance (bytes per triangle and bytes per vertex).

## Installation

To use MeshPress, clone the repository and install any dependencies. The project is compatible with Python 3.x and requires standard scientific libraries (e.g., NumPy).

```bash
git clone https://github.com/maletsden/meshpress.git
cd meshpress
pip install -r requirements.txt
```

## Compression Models

The following table summarizes the statistics for each compression model:


| Encoder Model                                     | Bytes per triangle | Bytes per vertex | Compression rate | 
|---------------------------------------------------|--------------------|------------------|------------------|
| **BaselineEncoder**                               | 18.05              | 35.82            | 1.00             |
| **SimpleQuantizator (no packing)**                | 11.60              | 23.03            | 1.56             | 
| **SimpleQuantizator (fixed packing)**             | **5.53**           | **10.98**        | **3.26**         | 
| **SimpleQuantizator (binary range partitioning)** | 5.69               | 11.29            | 3.17             |
| **SimpleQuantizator (radix binary tree)**         | 5.59               | 11.09            | 3.22             | 

Each encoder model applies a different strategy for compressing 3D mesh data, allowing users to balance compression rate and computational complexity according to their needs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have improvements or suggestions.
