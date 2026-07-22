# Benchmarks

Component of `blacknode-cuda`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="benchmarks", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.benchmarks]
    nodes = ["components/benchmarks/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
