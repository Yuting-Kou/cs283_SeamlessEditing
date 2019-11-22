# cs283_SeamlessEditing
Yuting Kou, Yizhou Wang

## Paper selection
[Blend two different images without seams using poisson image editing.](https://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)

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
