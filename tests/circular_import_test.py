import contextlib
import importlib
import inspect
import pkgutil
import random
import sys
from types import ModuleType


def import_first_function(module_name: str):
    """This is a helper which imports the first function from a module.

    Performs: `from sbi.module_name import first_function_name`

    Args:
        module_name: Name of the module.
    """

    module = importlib.import_module(module_name)
    functions = inspect.getmembers(module, inspect.isfunction)

    if functions:
        first_function_name, first_function = functions[0]
        cmd = f"from {module_name} import {first_function_name}"
        cmd += f"; del {first_function_name}"
        exec(cmd)
    else:
        pass


def reset_environment():
    """This is a helper which resets the environment by deleting all sbi modules."""
    # NOTE: Beware when working with sys.modules directly as this can lead to side
    # effects, e.g., sporadic errors with pickle!
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("sbi"):
            if module_name in globals():
                del globals()[module_name]
            elif module_name in locals():
                del locals()[module_name]


def find_submodules(package_name: str):
    """This is a helper which finds all submodules of a package.

    Args:
        package_name: Name of the package.
    """
    submodules = []
    package = __import__(package_name)

    def walk_submodules(package: ModuleType):
        """This is a recursive helper function which walks all submodules of a package.

        Args:
            package: The package to crawl through.
        """
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


def test_for_circular_imports():
    """This test checks for circular imports in the sbi package.

    In order to do so, it is tested if we can directly import from all submodules i.e.
    `from sbi.module_name import first_function_name` or `import sbi.module_name`
    without any import errors.

    """
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
            # NOTE: There might be other errors which are intended
            if "circular import" in str(e):
                # This is a circular import detected
                errors.append(f"Circular import detected in {module_name}. Error: {e}")
                print(f"Circular import detected in {module_name}")
        except Exception as e:
            # NOTE: There might be other errors which we catch here and report
            print(f"Error in {module_name}: {e}")
            errors.append(f"Error in {module_name}: {e}")

    assert len(errors) == 0, "\n".join(errors)
