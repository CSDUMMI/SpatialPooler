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
in the `.ipynb` files.
