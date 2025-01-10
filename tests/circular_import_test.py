import contextlib
import importlib
import inspect
import pkgutil
import random
import sys


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

    # Now crawls all submodules
    def walk_submodules(package):
        for _, name, is_pkg in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            submodules.append(name)
            if is_pkg:
                # There are some wanted import errors for deprecated modules
                with contextlib.suppress(ImportError):
                    walk_submodules(importlib.import_module(name))

    walk_submodules(package)
    return submodules


def reset_environment():
    # This is a helper which resets the environment
    for module_name in list(sys.modules.keys()):
        if "sbi" in module_name:
            del sys.modules[module_name]


def test_for_circular_imports():
    modules = find_submodules("sbi")
    # Permute the list of modules
    random.shuffle(modules)

    errors = []
    for module_name in modules:
        # Try to import
        if "sbi.examples" in module_name:
            # This is not really a module :/ Hence skip it...
            continue
        try:
            reset_environment()
            # Tests if we can: import module_name
            module = importlib.import_module(module_name)
            reset_environment()
            # Tests if we can: from module_name import xxx
            import_first_function(module.__name__)

            del module
        except ImportError as e:
            if "circular import" in str(e):
                errors.append(f"Circular import detected in {module_name}. Error: {e}")
                print(f"Circular import detected in {module_name}")

    assert len(errors) == 0, "\n".join(errors)
