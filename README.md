# Image Super-Resolution Using Deep Convolutional Networks

## From Abstract
### 1.  This report deep CNN
 - unlike traditional methods(handle each component separately), jointly optimizes all layers
 - lightweight structure, 
 - achieves fast speed for practical on-line usage

This report explore different network structures and parameter settings to achieve tradeoffs between performance and speed. 
This method extend the network to cope with three color channels simultaneously, and show better overall reconstruction quality.


---- 

## From INTRODUCTION

1. image super-resolution (SR) : aims at recovering a high-resolution image from a single lowresolution image
 - prior soluation: example-based strategy
    - ideas of prior soluation: example-based strategy methods
       - exploit internal similarities of the same image
       - learn mapping functions from external low- and high-resolution exemplar pairs
    - example-based strategy can formulated in
      - generic image super-resolution, 
      - can be designed to suit domain specific tasks, i.e., face hallucination
 
    - The sparse-coding-based method: one of the representative external example-based SR methods
      1. First, overlapping patches are densely cropped from the input image and pre-processed 
      2. These patches are then encoded by a low-resolution dictionary
      3. The sparse coefficients are passed into a high-resolution dictionary for reconstructing high-resolution patches.
      4. The overlapping reconstructed patches are aggregated e.g., by weighted averaging
      
      - This pipeline is shared by most external example-based methods, which pay particular attention to learning and optimizing the dictionaries  or building efficient mapping functions 
      - ### However, the rest of the steps in the pipeline have been rarely optimized or considered in an unified optimization framework.

2. This report method differs fundamentally from existing external example-based approaches
  1. This method does not explicitly learn the dictionaries or manifolds. for modeling the patch space
  2. These are implicitly achieved via hidden layers.
  3. the patch extraction and aggregation are also formulated as convolutional layers
  4. the entire SR pipeline is fully obtained through learning, with little pre/postprocessing
  - Named: Super-Resolution Convolutional Neural Network
- appealing properties
  1. its structure is intentionally designed with simplicity in mind and yet provides superior accuracy compared with state-of-the-art example-based methods(  Figure 1 shows a comparison on an example.)
  2. with moderate numbers of filters and layers, our method achieves fast speed for practical on-line usage even on a CPU.(faster than a number of example-based methods because it is fully feed-forward and does not need to solve any optimization problem on usage)
  3.  experiments show that the restoration quality of the network can be further improved when (i) larger and more diverse datasets are available, and/or (ii) a larger and deeper model is used. (######### On the contrary, larger datasets/models can present challenges for existing example-based methods. Furthermore, the proposed network can cope with three channels of color images simultaneously to achieve improved super-resolution performance########)
  
3. The contributions of this study
  1. present a fully convolutional neural network for image super-resolution. The network directly learns an end-to-end mapping between lowand high-resolution images, **with little pre/postprocessing beyond the optimization**
  2. establish a relationship between our deep learning-based SR method and the traditional sparse-coding-based SR methods. This relationship **provides a guidance for the design of the network structure.**
  3. **demonstrate that deep learning is useful in the classical computer vision problem of superresolution, and can achieve good quality and speed**


  
  ## From RELATED WORK
  
  Not read
  
  
  
##  From CONVOLUTIONAL NEURAL NETWORKS FOR SUPER-RESOLUTION
  
  
  
  
  
  
  
  
  
  
  
  
