# CoRM-PPM
This repository contains the code and data for the CoRM-PPM presented in Chapter 5. The C++ source code can be found in the folder src. 

## 🛠️ Compiler Requirement

If you wish to re-run the MCMC from scratch, a **g++ compiler** is required to build and run the C++ code. 

### Installation Instructions

- **macOS**:  
  Install via [Homebrew](https://brew.sh/):
  ```bash
  brew install gcc

- **Windows**  
  Install [MinGW-w64](https://www.mingw-w64.org/) or use [MSYS2](https://www.msys2.org/).
  After installation, ensure g++ is added to your system PATH.
  You can verify it by running:
  ```bash
  g++ --version


The notebooks "eeg_analysis.ipynb" and "sim_study_analysis.ipynb" contain instructions for reproducing the plots and figures for the real EEG data example and simulation study. The simulated data can be found in the SimData folder.
