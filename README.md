# A Deep Learning-based Method for Estimating Human Weight and Height from Multi-view Clinic Images

This repository contains the algorithm developed for estimating human weight and height using multi-view images. The code supports training both Wide ResNet and Scale-Equivariant WideResnet models.

![HEADER]()

## Table of Contents
- [📖 Introduction](#-introduction)
- [🛠️ Tools](#️-tools)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [✨ Features](#-features)
- [📦 Dependencies](#-dependencies)
- [📝 Configuration](#-configuration)
- [🔍 Examples](#-examples)
- [❓ Troubleshooting](#-troubleshooting)
- [👥 Contributors](#-contributors)
- [📜 License](#-license)
- [📧 Contact](#-contact)

## 📖 Introduction
This project presents a deep learning approach to estimate human weight and height from clinic images taken from multiple views. Leveraging advanced neural network architectures, this method provides accurate and reliable estimations, potentially aiding various clinical and health-related applications.

## 🛠️ Tools
- Python 3.10
- Cuda 11.7
- Conda 23.3

## ⚙️ Installation
1. **Download and Install Anaconda**  
   Visit the [Anaconda website](https://www.anaconda.com/about-us) to download and install Anaconda.

2. **Clone the Repository**  
   Clone the project repository from GitHub:
   ```sh
   git clone https://github.com/lucasdblucas/wh_model.git
   cd wh_model
   ```

3. **Create a Virtual Environment**  
   Create a virtual environment using the provided `environment.yml` file:
   ```sh
   conda env create -f environment.yml
   ```

4. **Activate the Conda Environment**  
   Activate the newly created environment:
   ```sh
   conda activate env_name
   ```

## 🚀 Usage
1. **Navigate to the Execution Folder**  
   Change directory to the `src/bashs` folder:
   ```sh
   cd src/bashs
   ```

2. **Run Experiments**  
   Execute the experiment scripts using the appropriate configuration files found in `src/config`.

   - **WideResnet Experiment:**
     ```sh
     bash execute_wh_21.sh
     ```
   - **Scale-Equivariant WideResnet Experiment:**
     ```sh
     bash execute_wh_22.sh
     ```

## ✨ Features
- Supports training of Wide ResNet and Scale-Equivariant WideResnet models.
- Utilizes multi-view clinic images for enhanced estimation accuracy.
- Configuration-driven execution for flexible experimentation.

## 📦 Dependencies
The project dependencies are managed using Conda and specified in the `environment.yml` file. Ensure to create and activate the conda environment as described in the installation steps.

## 📝 Configuration
The configuration files for running experiments are located in the `src/config` folder. These files define the parameters and settings used during training and evaluation.

## 🔍 Examples
Examples of running the experiments are provided in the usage section. Additional example configurations and results can be added to this section as needed.

## ❓ Troubleshooting
For common issues and troubleshooting steps, please open an issue on the project's GitHub page.

## 👥 Contributors
* Lucas Daniel Batista Lima
* Ariel Teles Soares

## 📜 License
- ### <a href="https://doi.org/10.1016/j.eswa.2024.124879"><img src="https://zenodo.org/badge/DOI/10.1016/j.eswa.2024.124879.svg" alt="DOI"></a> 

## 📧 Contact
For any questions or inquiries, please contact:
- ![Gmail Badge](https://img.shields.io/badge/-lucasbatista@ufdpar.edu.br-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:ariel.teles@ifma.edu.br) ![Gmail Badge](https://img.shields.io/badge/-lucas.daniel.bp@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:lucas.daniel.bp@gmail.com)
- ![Gmail Badge](https://img.shields.io/badge/-ariel.teles@ifma.edu.br-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:ariel.teles@ifma.edu.br)
---

Feel free to contribute to the project or reach out with any suggestions or improvements.