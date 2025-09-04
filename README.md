# Interactive Image Generation using GANs
A GAN-based framework for sketch-to-image translation, style transfer, and interactive image manipulation with dual implementations for image manipulation: cGANs, DCGANs, and an OpenCV UI for real-time editing.

This project is our implementation of the interactive image generation techniques inspired by the seminal work:  
**“Generative Visual Manipulation on the Natural Image Manifold”** by Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, and Alexei A. Efros (ECCV, 2016).

While the original work provided the core idea, our journey involved building this system from the ground up using PyTorch and OpenCV, training on specific datasets, and tackling the unique challenges of real-time user interaction.
## Interactive UI
<p align="center">
  <img src="https://github.com/ShashwatiBuragohain/Generative-Visual-Manipulation-on-the-Natural-Image-Manifold/blob/ebd077adaf6914050d3048ad49397d9201a2548e/images/Interactive%20UI.png?raw=true" width="60%" />
</p>


*This is our interactive OpenCV interface. Here, after uploading the image we scribble on it using the color brush and then the edited image gets displayed after applying the gan edit option as shown below*

<p align="center">
  <img src="https://github.com/ShashwatiBuragohain/Generative-Visual-Manipulation-on-the-Natural-Image-Manifold/blob/ebd077adaf6914050d3048ad49397d9201a2548e/images/scribble1.jpg?raw=true" width="25%" />
  <img src="https://github.com/ShashwatiBuragohain/Generative-Visual-Manipulation-on-the-Natural-Image-Manifold/blob/ebd077adaf6914050d3048ad49397d9201a2548e/images/scribble2.jpg?raw=true" width="25%" />
</p>

<p align="center">
  <b>Left:</b> Input UI (scribbled image) &nbsp;&nbsp;&nbsp; <b>Right:</b> Generated Image
</p>

## Guided Manipulation

This process is the heart of "Generative Visual Manipulation on the Natural Image Manifold." We took a generated shoe and modified its edge map to give it a higher top, altering its fundamental structure.

We started with a generated image and its corresponding edge map.

- Using our interactive canvas, we directly edited the edge map, extending the lines to sketch a new, higher shoe profile.

- This new, "out-of-distribution" edge map was then projected onto the learned manifold using our optimization procedure (minimizing L1 and TV loss).

- The generator took this optimized, plausible edge map and produced a new, coherent image that closely followed the user's structural edit.

The result, as shown below, demonstrates this approach. The system successfully incorporates the user's guidance—a higher top—while generating photorealistic textures and details that are consistent with the new structure.

<p align="center">
  <img src="https://github.com/ShashwatiBuragohain/Generative-Visual-Manipulation-on-the-Natural-Image-Manifold/blob/ebd077adaf6914050d3048ad49397d9201a2548e/images/style%20transfer.jpg?raw=true" width="50%" />
</p>

Direct Manipulation via Edge Editing: The input edge map (left) was modified by a user. Our system projected this edit onto the natural image manifold and generated a new, coherent image (right) that incorporates the change while maintaining realism.

## DATASET:

We used two of the datasets mentioned in the original paper:
###  1. Paired Datasets (for cGAN / Pix2Pix)

- **edges2shoes.tar.gz** (2.0 GB)
- **edges2handbags.tar.gz** (8.0 GB)  
   [Download from Pix2Pix Datasets](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)

Since these two dataset contained paired images(original+edge) we used cGAN(conditional GAN) and used them for image-to-image translation and style transfer. 
For the implementation of scribble to image generation we wanted a more diverse dataset. 
###  2. Unpaired Dataset (for DCGAN)

To implement **scribble-to-image generation**, we needed a more diverse dataset. So, we used:

- **Mountains and Beaches Dataset** from Kaggle  
   [Download from Kaggle](https://www.kaggle.com/datasets/erennik/mountains-and-beaches-dataset)

This dataset was trained using **DCGAN**, as it doesn't contain paired samples like the others.

## MODELS:
We decided to implement two distinct architectures to understand their strengths.
### Pix2Pix (cGAN with U-Net + PatchGAN)

- **Generator**: U-Net (built from scratch with skip connections)
- **Discriminator**: PatchGAN
- **Datasets Used**: edges2shoes & edges2handbags

The U-Net preserved sketch details, and PatchGAN focused on **local realism**. 

Here, we built the U-Net generator from scratch. Its skip connections were crucial for preserving the spatial details of the user's sketch. We paired it with a PatchGAN discriminator to ensure the generated images were sharp and focused on local realism. This model became the core of our sketch-to-image translation system, trained on the Edges2Shoes and Edges2Handbags datasets .

###  DCGAN (for Scene Generation from Noise)

- Used for generating natural scenes from random vectors
- Trained on the **Mountains and Beaches (Places) Dataset**
- Helped us understand **unconditional generation** and the challenges of diverse data

In parallel to the cGAN, we developed a DCGAN to learn the structure of complex outdoor scenes from the Places dataset. Here we were highlighted the challenges of training GANs on diverse data and demonstrated the power of a latent space for generating images from pure noise.
## Bridging the Gap (The Algorithm):

The paper's key insight was **manifold projection**—finding the best input (a sketch or latent vector) that makes the generator output a realistic image matching the user's edits.

We implemented this optimization process. When a user provides a sketch, the system doesn't just run it through the generator. It actively refines the sketch to lie on the "natural image manifold" by minimizing a custom loss function:

L = ||G(E) - Target||₁ + λ_tv * L_TV(E)


This **L1 Loss** ensures the output matches the user's intent, while the **Total Variation (TV) Loss** we added acts as a regularizer, smoothing out noisy strokes to create cleaner, more realistic edges for the generator to work with.

## The Interface
A powerful backend is useless without a way to talk to it. This led to the most rewarding phase: building the UI.

Using OpenCV, we created a simple canvas. The process was iterative:

- We started with a basic window where you could draw lines.
- We added Gaussian smoothing to the brush strokes on the fly, which dramatically improved output quality by reducing jagged edges.

## Features

- Real-Time Sketch-to-Image Translation: Draw an edge map and generate a corresponding realistic image instantly.
- Interactive OpenCV Canvas: An intuitive interface for drawing and editing with real-time visual feedback.
- Dual GAN Implementations: Includes both a cGAN (Pix2Pix) for conditional generation and a DCGAN for unconditional scene generation.
- Manifold Projection: Utilizes optimization techniques to project user inputs onto the natural image manifold for higher quality results.


## Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- NumPy

## Citation

```bibtex
@inproceedings{zhu2016generative,
  title={Generative Visual Manipulation on the Natural Image Manifold},
  author={Zhu, Jun-Yan and Kr{\"a}henb{\"u}hl, Philipp and Shechtman, Eli and Efros, Alexei A.},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2016},
  doi={10.1007/978-3-319-46454-1_36}
}
``` 
## Authors & Acknowledgments

Built by: 
[Shashwati Buragohain] & [Swagata Buragohain]  
This project was completed as part of the Advanced Machine Learning (EE 525) course at Indian Institute of Technology Guwahati (IITG).

We extend our gratitude to the authors of the original paper for their inspiring work.
