{{ objname }}
{{ underline }}

{# `:inherited-members:` matches on the bare class name, so `torch.nn.Module` would
   silently match nothing. The excluded names are annotated attributes of `nn.Module`,
   which the filter does not drop reliably (e.g. Sphinx 8.1 on Python 3.10). #}
.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :inherited-members: Module
   :exclude-members: training, call_super_init, dump_patches
   :show-inheritance:
