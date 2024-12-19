from datetime import datetime
import pytest

import shutil


def pytest_addoption(parser):
    parser.addoption(
        "--print-harvest",
        action="store_true",
        default=False,
        help="Print the harvest results at the end of the test session",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Only print the harvest results if the --print-harvest flag is used
    if config.getoption("--print-harvest"):
        # Dynamically center the summary title in the terminal width
        terminal_width = shutil.get_terminal_size().columns
        summary_text = " short test summary info "
        centered_line = summary_text.center(terminal_width, '=')
        terminalreporter.write_line(centered_line)


@pytest.mark.parametrize('p', ['world', 'self'], ids=str)
def test_foo(p, results_bag):
    """
    A dummy test, parametrized so that it is executed twice
    """

    # Let's store some things in the results bag
    results_bag.nb_letters = len(p)
    results_bag.current_time = datetime.now().isoformat()


def test_synthesis(fixture_store):
    """
    In this test we inspect the contents of the fixture store so far, and
    check that the 'results_bag' entry contains a dict <test_id>: <results_bag>
    """
    # print the keys in the store
    results = fixture_store["results_bag"]

    # print what is available for the 'results_bag' entry
    print("\n--- Harvested Test Results ---")
    for k, v in results.items():
        print(k)
        for kk, vv in v.items():
            print(kk, vv)
