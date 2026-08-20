# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import threading

from sbi.utils.pbar import is_nested, nested_pbar_context


class TestNestedPbarContext:
    def test_not_nested_by_default(self):
        assert not is_nested()

    def test_nested_inside_context(self):
        with nested_pbar_context():
            assert is_nested()

    def test_not_nested_after_context_exit(self):
        with nested_pbar_context():
            pass
        assert not is_nested()

    def test_deeply_nested_contexts(self):
        with nested_pbar_context():
            assert is_nested()
            with nested_pbar_context():
                assert is_nested()
                with nested_pbar_context():
                    assert is_nested()
                assert is_nested()
            assert is_nested()
        assert not is_nested()

    def test_thread_isolation(self):
        main_sees = []
        worker_sees = []
        barrier = threading.Barrier(2, timeout=5)

        def worker():
            with nested_pbar_context():
                barrier.wait()
                worker_sees.append(is_nested())
                barrier.wait()

        t = threading.Thread(target=worker)
        t.start()
        barrier.wait()
        main_sees.append(is_nested())
        barrier.wait()
        t.join()

        assert not main_sees[0], "main should not see worker's context"
        assert worker_sees[0], "worker should see its own context"

    def test_independent_contexts(self):
        with nested_pbar_context():
            first = is_nested()
        with nested_pbar_context():
            second = is_nested()
        assert first and second
