# cs283_SeamlessEditing
Yuting Kou, Yizhou Wang

## Reference paper 
[Blend two different images without seams using poisson image editing.](https://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)

This paper proposed a generic framework of guided interpolation to achieve seamless editing. Given a properly-chosen guidance field  <img src="https://latex.codecogs.com/gif.latex?v" />, this paper solves Poisson equation to find best interpolant function <img src="https://latex.codecogs.com/gif.latex?f" /> which minimizes the error with guidance vector field around the corners. 

<img src="https://latex.codecogs.com/gif.latex?\min_f\int\int_\Omega&space;|\nabla&space;f-v|^2&space;\text{&space;with&space;}&space;f|_{\partial&space;\Omega}=f^*|_\partial&space;\Omega" title="\min_f\int\int_\Omega |\nabla f-v|^2 \text{ with } f|_{\partial \Omega}=f^*|_\partial \Omega" />
<img src="https://latex.codecogs.com/gif.latex?\Leftrightarrow\Delta&space;f=\text{div}&space;v&space;\text{&space;with&space;}&space;f|_{\partial&space;\Omega}=f^*|_\partial&space;\Omega" title="\Leftrightarrow\Delta f=\text{div} v \text{ with } f|_{\partial \Omega}=f^*|_\partial \Omega" />

By choosing different guidance field, we can achieve different functional tools for seamless editing:
- Seamless cloning: (related to source img: merge two img)
    - <img src="https://latex.codecogs.com/gif.latex?v=\nabla&space;g" title="v=\nabla g" /> : insert source picture <img src="https://latex.codecogs.com/gif.latex?g"/> into destination picture <img src="https://latex.codecogs.com/gif.latex?f^*" />.
    - <img src="https://latex.codecogs.com/gif.latex?v=\begin{cases}\nabla&space;f^*(x)&space;&&space;\text{if}&space;|\nabla&space;f^*(x)|>|\nabla&space;g(x)|\\\nabla&space;g(x)&&space;\text{otherwise}\end{cases}" title="v=\begin{cases}\nabla f^*(x) & \text{if} |\nabla f^*(x)|>|\nabla g(x)|\\\nabla g(x)& \text{otherwise}\end{cases}" />: insert objects with holes or partially transparent objects on top of some textured background.
- Selecting editing: (related to destination img: in-place img transformation)
    - <img src="https://latex.codecogs.com/gif.latex?v&space;=&space;M(x)\nabla&space;f^*(x)" title="v = M(x)\nabla f^*(x)" /> : texture flattening: apply a sparse matrix filter over gradient of destination image.
    - local illumination changes: apply non-linear transformation (e.g. log) to the gradient field, then integrating back with a Poisson solver.
    - local color change: set destination image to greyscale, solve Poisson equation.
    - seamless tilting: set periodic boundary values on the borader of a rectangular region before integrating with Poisson solver.
## Paper Abstract

Poisson Image Editing is a framework for seamless image editing and yields a variety of novel tools. Based on solving the Poisson equation with Dirichlet boundary condition, a function can be found which smoothly connects the destination image and source image, and optimally preserves the gradient information in the target area. However, gradients only retain the relative information of color and illumination inside the target region. The shift of pixel values actually depends on the boundary conditions. This is the key to the algorithm's success in seamless editing, but it also results in certain limitation -- the method is unable to seamlessly mix two pictures with very different illumination conditions while ensuring that the objects in the interior of the target area still look natural. In this paper, we discover this limitation through experiments and proposed several potential solutions for future research.
## User Guidance

For a user guide on how to use our code and replicate the results in a PDF report using code, see  [Example.ipynb](Example.ipynb).
