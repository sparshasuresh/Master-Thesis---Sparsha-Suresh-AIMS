# Master Thesis – Sensor Fusion Prototype (Sparsha Suresh)

This repository contains the implementation code and supporting materials for my Master's thesis on **sensor fusion for autonomous vehicles**, demonstrated in the CARLA simulator. The prototype integrates GNSS, IMU, and LiDAR data using a Kalman filter, with a Bayesian reliability model that adapts to changing weather conditions.

---

## 🚀 Installation

### Requirements
- **Operating System:** Ubuntu 20.04 (recommended)
- **Python:** 3.8
- **CARLA Simulator:** Version 0.9.13 ([CARLA Installation Guide](https://carla.readthedocs.io/en/latest/start_quickstart/))
- **Dependencies:**
  ```bash
  pip install numpy

  # Software Installation Guide for CARLA + Python Virtual Environment

---

## Software Installation
![alt text](image.png)

### 📌 Figure: Clone the CARLA Repository
![alt text](image-1.png)
### 📌 Figure: Install Dependencies

![alt text](image-2.png)
### 📌 Figure: Build CARLA from Source
![alt text](image-3.png)
### 📌 Figure: Run CARLA Simulator


CARLA (Car Learning to Act) is created to support research and advancement in autonomous vehicle systems. Its realistic environment which includes dynamic actors, sensor reproduction, roadways and weather, makes testing and verifying autonomous driving code in a controlled way very convenient (Catozzi, 2024, pg: 81).

On Ubuntu 22.04 LTS, first clone the official CARLA repo from GitHub with the `git clone` command. It gets the most up-to-date source code and simulation assets to local computer. Once you go into the `carla` directory with `cd carla`, the following step is to install the dependencies which consist of compilers, build tools (including cmake and clang), image processing libraries (such as libpng, libtiff, libjpeg) and Python packages for development (Činčurak et al., 2024, pg: 139).

---

Once you have all dependencies, you need to use `make PythonAPI` and `make launch` from the CARLA source to start building. Assembling the code in this way makes certain everything on the local computer is set up properly to run the simulator (Činčurak et al., 2024, pg: 111). With the Python API, developers can add their own actions to a simulation and connect algorithms like sensor fusion.

---

Once the build is over, start CARLA by typing ./CarlaUE4.sh. Should the installation be successful, a graphical simulation window will appear with a world where you can instantiate and maneuver automated cars. Because it is working properly, the simulator is ready to be integrated with ROS2 and custom Python AI modules (Goyal, Griggio and Tonetta, 2025, pg: 36). 
---

## Create a Python Virtual Environment
![alt text](image-8.png)
### 📌 Figure: Install Miniconda

![alt text](image-9.png)
### 📌 Figure: Create and Activate Virtual Environment

![alt text](image-10.png)
### 📌 Figure: Install Required Python Packages

---

Establishing a special Python virtual environment is necessary for using software like CARLA and Scenario Runner which could have conflicting requirements. Using a virtual environment, each tool is set up separately, so that it does not conflict with system-level packages (Gutierrez et al., 2021, pg: 84).

The last step is to use pip to set up needed Python libraries. Among these, numpy is used for handling arrays, pandas is used for analysis, opencv-python is used for displaying images. It makes sure that the work area for development is always organized (Jeon et al., 2022, pg: 65).





