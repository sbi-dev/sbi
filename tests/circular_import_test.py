import importlib
import inspect
import pkgutil
import random


def import_first_function(module_name: str):
    # This is a helper which emulates the following:
    # from sbi.xxx import yyy
    module = importlib.import_module(module_name)
    functions = inspect.getmembers(module, inspect.isfunction)

    if functions:
        first_function_name, first_function = functions[0]
        cmd = f"from {module_name} import {first_function_name}"
        exec(cmd)
        cmd_del = f"del {first_function_name}"
        exec(cmd_del)
    else:
        pass


def find_submodules(package_name):
    # This is a helper which finds all modules from which we could import
    submodules = []
    package = __import__(package_name)

    for _, name, _ in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        submodules.append(full_name)

    return submodules


def test_for_circular_imports():
    modules = find_submodules("sbi")
    # Permute the list of modules
    random.shuffle(modules)

    for module_name in modules:
        # Try to import
        if "sbi.examples" in module_name:
            # This is not really a module :/ Hence skip it...
            continue
        try:
            # Tests if we can: import module_name
            module = importlib.import_module(module_name)
            # Tests if we can: from module_name import xxx
            import_first_function(module.__name__)
            del module
        except ImportError as e:
            raise AssertionError(f"Cannot import {module_name}. Error: {e}") from e
