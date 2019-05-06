# Spatial Pooler - Simple and minimal implementation of the Spatial Pooler by Numenta

If you want a deep dive into the
theory behind the Spatial Pooler:

**HTM School:**
https://invidio.us/watch?v=R5UoFNtv5AU

https://invidio.us/watch?v=rHvjykCIrZM

**Numenta Homepage(s):**
https://numenta.com/
https://numenta.org/

**Blog post:**
https://numenta.org/blog/2016/02/19/mathmatical-formalization-of-htm-spatial-pooler/

# How to use this
If you are in this Folder:
```python3
from spatial_pooler import SpatialPooler, overlap
import numpy as np

sp = SpatialPooler(50,100)
output  = sp.run(np.random.randn(50) > 0)

print(output)
```

The most important
method in the
`SpatialPooler` class
is `SpatialPooler.run()`.
It takes a numpy bool array,
that represents a SDR and
returns another numpy bool
array that is the output
of the SP.

#### The SP always learns.

More examples and documentation
in the `.ipynb` files in the
`examples` folders.


## The Architecure and meaning of State -> Potential Pool -> Permanence -> Activation -> State'
As you may have noted, the description
of this Repository is State -> Potential Pool -> Permanence -> Activation -> State',
but what does this mean?
If your are more into SP Theory, you may noted that this is
a picture of the SP Algorithm.
First there is a State, a SDR that is
also the input to the SP.
And then for each column in the SP,
the Algorithm first ignores those
parts of the state that are not in
the potential pool, then those
for which the permanence is less
than a threshhold.
After these two extractions,
the algorithm counts the active
parts of the state left and
tests again whether that count
is greater than a threshhold.
If so, the collumn upon which these
steps where executed is activated
and the output is activated.
