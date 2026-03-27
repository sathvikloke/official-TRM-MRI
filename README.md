# TRM-MRI

TRM-MRI is a standalone MRI reconstruction framework based on the [Tiny Recursive Models (TRM)](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) developed by SamsungSAILMontreal. This project adapts the recursive TRM architecture for MRI k-space reconstruction, enabling small, fast, and accurate models with physics-based iterative refinement.

## Features

- Recursive refinement of MRI images with shared weights  
- Lightweight TRM-based architecture for small model size  
- Physics-based data consistency using FFTs  
- Support for continuous-valued MRI outputs  
- MSE / PSNR evaluation metrics  
- Configurable recursion steps for speed vs. quality tradeoff  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sathvikloke/official-TRM-MRI.git
cd TRM-MRI
