# GestaltRACS-UAV

Gestalt-based region-adaptive compressive sensing for efficient and long-term UAV infrared imaging.

## Description

This project implements a Gestalt-based region-adaptive compressive sensing (RACS) approach tailored for efficient and long-term infrared (IR) imaging using Unmanned Aerial Vehicles (UAVs). The core idea is to leverage Gestalt principles to intelligently divide IR images into regions and then apply compressive sensing techniques adaptively based on the characteristics of each region. This allows for significant data reduction while preserving essential information, enabling longer flight times and more efficient data storage and transmission.

## Installation

### Prerequisites

*   Python 3.6+
*   [PyTorch](https://pytorch.org/) (tested with version >= 1.0)
*   NumPy
*   SciPy
*   OpenCV
*   Other dependencies (install using `pip install -r requirements.txt` after cloning the repository, if a `requirements.txt` file exists.  If not, install the following packages: `scikit-image`, `matplotlib`)

### Installation Steps

1.  Clone the repository:

    ```bash
    git clone https://github.com/historier/GestaltRACS-UAV.git
    cd GestaltRACS-UAV
    ```

2.  (Optional) Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  Install the required packages:

    ```bash
    pip install numpy scipy opencv-python scikit-image matplotlib torch torchvision
    ```

## Key Features

*   **Gestalt-based Region Segmentation:** Employs Gestalt principles to intelligently segment IR images into meaningful regions.  This is implemented in `gestalt_ir_image_division.py`.
*   **Region-Adaptive Compressive Sensing:** Applies compressive sensing techniques adaptively to each region based on its characteristics.
*   **Reconstruction Network (ReconNet):** Includes a reconstruction network (`reconnet_train.py`) for recovering the original image from the compressed data.
*   **Data Preprocessing:** Provides scripts for preprocessing the HIT-UAV-Infrared-Thermal Dataset (`data_pre01.py`, `data_pre02.py`, `data_pre03.py`, `data_pre04.py`).
*   **Dataset Splitting:** Includes a script for splitting the dataset into training and testing sets (`dataset_split.py`).
*   **Visualization Tools:** Offers tools for visualizing the data and results (`data_visualization.py`).
*   **Demo:** A demo script (`demo.py`) to showcase the functionality of the project.

## Contribution Guidelines

Contributions are welcome! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, concise messages.
4.  Test your changes thoroughly.
5.  Submit a pull request.

## Contributors

*   historier
```