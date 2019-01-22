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

1. Formulation
  1. Only pre-processing: upscale low-resolution image to the desired size(method: using bicubic interpolation)
  2. wish to learn a mapping F(from "low-resolution" image to high-resolution image), conceptually consists of three operations
      1. Patch extraction and representation：该操作从低分辨率图像Y中提取（重叠）块，并将每个块表示为高维矢量。 这些向量包括一组特征图，其数量等于向量的维数。
      2. this operation nonlinearly maps each high-dimensional vector onto another high-dimensional vector. comprise another set of feature maps
      3. Reconstruction: this operation aggregates the above high-resolution patch-wise representations to generate the final high-resolution image.
  
  
  3.1.1 Patch extraction and representation
  3.1.2 Non-linear mapping
  3.1.3 Reconstruction
  3.2 Relationship to Sparse-Coding-Based Methods
  Not read
  
##  3.3 Training
1. Given a set of high-resolution images {Xi} and their corresponding low-resolution images {Yi}, we use **Mean Squared Error (MSE)** as the loss function.(reason of using MSE:  Using MSE as the loss function favors a high PSNR. The PSNR is a widely-used metric for quantitatively evaluating image restoration quality, and is at least partially related to the perceptual quality. )
2. 

 
  
  
  
  
  
  
  
  
  
  
