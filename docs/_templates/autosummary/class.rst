{{ objname }}
{{ underline }}

{# `:inherited-members:` matches on the bare class name, so `torch.nn.Module` would
   silently match nothing. #}
.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :inherited-members: Module
   :exclude-members: training
   :show-inheritance:
