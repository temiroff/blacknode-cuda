# Image Processing

Component of `blacknode-cuda`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="image-processing", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.image-processing]
    nodes = ["components/image-processing/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
