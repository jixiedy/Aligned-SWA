#from __future__ import absolute_import

# import os
# import sys

# path = os.path.dirname(os.path.abspath(__file__))
# print('path', path)
# print('f_list', [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py'])
# for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
#     mod = __import__('.'.join([__name__, py]), fromlist=[py])
#     print('mod', mod)
#     classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
#     print('classes', classes)
#     for cls in classes:
#         setattr(sys.modules[__name__], cls.__name__, cls)
#         print('setattr', setattr)
