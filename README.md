# cs283_SeamlessEditing
Yuting Kou, Yizhou Wang

## Paper selection
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
## Research Schedule
### Paper extension
In this paper, a variety of novel tools are introduced for the seamless editing of image regions, using generic interpolation machinery based on solving Poisson equations. Here we propose several possible extensions. During our research process, we may implement some or all of these ideas based on the feasibility.

### Extension over algorithms
- Seamless cloning: figure out more choices for the guidance field. The paper offered to use a linear and very simple non-linear combinations of source and destination gradient field. We can explore more complex types of non-linear mixing of gradient fields.
- Selection editing: try other filters on image gradient to achieve different results of textual transformation. The paper proposed to use a sparse sieve that retains only the most salient features to do textual flattening.

### Extension over applications
- Object insertion:
    - Seamless insert people into their idol’s picture
    - Seamless clone people onto landscape picture or fancy background
- Object replacement:
    - Seamless replace face: E.g. can be used in the barber shop to see if a new hairstyle fits the customer.
    - Modify the expression of people in photographs. E.g. make people smile; replace closed eyes with open eyes
- Concealment:
    - Beautify: remove spot/acne, remove wrinkle
    
### Potential dataset
- Face dataset:
    - [Face recognition database](http://www.face-rec.org/databases/)
    - [Harvard face database](http://vision.seas.harvard.edu/pubfig83/)
- Other sources:
    - Our own pictures
    - Google pictures 
