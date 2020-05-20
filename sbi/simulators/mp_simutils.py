import multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm
from typing import Callable, Tuple, List
import torch
from sys import platform
from torch import Tensor
from importlib import reload
import logging


def simulate_mp(
    simulator: Callable,
    theta: Tensor,
    num_workers: int = 4,
    worker_batch_size: int = 20,
    show_progressbar: bool = True,
    logging_level: int = logging.WARNING,
) -> Tuple[Tensor, Tensor]:
    """
    Return parameters theta and simulated data x, simulated using multiprocessing on the
        passed simulator.

    Args:
        simulator: Simulator function that will be executed on each worker.
        theta: All simulation parameters.
        num_workers: Number of parallel workers to start.
        worker_batch_size: Number of parameter sets that are processed per worker. Needs
            to be larger than sim_batch_size. A lower value creates overhead from
            starting workers frequently. A higher value leads to the simulation
            progressbar being updated less frequently (updates only happen after a
            worker is finished).
        show_progressbar: Whether to show a progressbar.
        logging_level: The logging level determines the amount of information printed to
            the user. Currently only used for multiprocessing. One of
            logging.[INFO|WARNING|DEBUG|ERROR|CRITICAL].

    Returns: parameters theta and simulation outputs x. The order of theta is not
        necessarily the same as for the input variable theta, which is why we return it
        here.
    """

    reload(logging)  # jupyter notebooks require reload to update the logging level
    # see here: SO 18786912
    logging.basicConfig(level=logging_level)

    queue, workers, pipes = start_workers(simulator=simulator, num_workers=num_workers)
    logging.info("Started workers")
    final_x = []  # will be filled with simulation outputs at job completion
    final_theta = []  # # will be filled with parameters theta at job completion
    minibatches = iterate_minibatches(theta, worker_batch_size)

    pbar = tqdm(
        total=len(theta),
        disable=not show_progressbar,
        desc=f"Running {len(theta)} simulations",
    )

    done = False
    with pbar:
        while not done:
            active_list = []
            for worker, pipe in zip(workers, pipes):
                try:
                    theta_worker_batch = next(minibatches)
                except StopIteration:
                    # stop passing data to workers
                    done = True
                    break

                active_list.append((worker, pipe))
                logging.info(f"Dispatching to worker (len = {len(theta_worker_batch)})")
                pipe.send(theta_worker_batch)
                logging.info("Done")

            num_remaining = len(active_list)
            while num_remaining > 0:
                logging.info("Listening to worker")
                msg = queue.get()
                if isinstance(msg, int):
                    logging.info("Received int")
                elif isinstance(msg, tuple):
                    logging.info("Received results")
                    x, theta = msg
                    final_x.append(x)
                    final_theta.append(theta)
                    num_remaining -= 1
                    pbar.update(len(theta))
                else:
                    logging.info(
                        f"Warning: Received unknown message of type {type(msg)}"
                    )

    stop_workers(workers, pipes, queue)

    # Concatenate to get shape (num_samples, shape_of_single_x).
    x = torch.cat(final_x, dim=0)
    # Concatenate to get shape (num_samples, shape_of_single_theta).
    theta = torch.cat(final_theta, dim=0)

    return theta, x


def start_workers(
    simulator: Callable, num_workers: int = 4
) -> Tuple[mp.Queue, List[mp.Process], List[mp.Pipe]]:
    """
    Start all workers.

    Args:
        simulator: Simulator function that will be executed on each core.
        num_workers: Number of parallel workers to start.

    Returns: queue, workers, pipes
    """
    try:
        if platform == "win32":
            # Windows requires spawn start method.
            mp.set_start_method("spawn")
        else:
            # Linux and macOS require fork start method.
            mp.set_start_method("fork")
    except RuntimeError:
        # try-except because start method can not be set multiple times within the same
        # session. So, if the user runs things outside of a __main__() function, e.g. in
        # a jupyter notebook, this would give an error from the second call of the
        # set_start_method() function on
        pass

    parents, children = zip(*(mp.Pipe(duplex=True) for _ in range(num_workers)))
    worker_cfg = dict(queue=mp.Queue(), simulator=simulator, seed=None)  # TODO #166
    workers = [Worker(i, pipe=child, **worker_cfg) for i, child in enumerate(children)]

    logging.info("Starting workers")
    for w in workers:
        w.start()

    logging.info("Done")

    return worker_cfg["queue"], workers, parents


def stop_workers(workers, pipes, queue) -> None:
    """
    Stop all workers.

    Args:
        workers: Workers.
        pipes: Pipes.
        queue: Queue.
    """
    if workers is None:
        return

    logging.info("Closing")
    for w, p in zip(workers, pipes):
        logging.info("Closing pipe")
        p.close()

    for w in workers:
        logging.info("Joining process")
        w.join(timeout=1)
        w.terminate()

    queue.close()


def iterate_minibatches(theta: Tensor, worker_batch_size: int = 20) -> Tensor:
    """
    Yields the thetas of shape [minibatch, shape_of_single_theta] from the entire
    parameter array theta

    Args:
        theta: All input parameters.
        worker_batch_size: Size of the minibatch to be passed to each worker.

    Returns: iterable where each entry is a minibatch of worker_batch_size thetas
    """
    num_samples = len(theta)

    for i in range(0, num_samples - worker_batch_size + 1, worker_batch_size):
        yield theta[i : i + worker_batch_size]

    rem_i = num_samples - (num_samples % worker_batch_size)
    if rem_i != num_samples:
        yield theta[rem_i:]


# todo: do we need seeding back? Or find if we just torch.manual_seed(0)?
# def reseed(self, seed):
#     """Carries out the following operations, in order:
#     1) Reseeds the master RNG for the generator object, using the input seed
#     2) Reseeds the prior from the master RNG. This may cause additional
#     distributions to be reseeded using the prior's RNG (e.g. if the prior is
#     a mixture)
#     3) Reseeds the simulator RNG, from the master RNG
#     4) Reseeds the proposal, if present
#     """
#     self.rng.seed(seed=seed)
#     self.seed = seed
#     self.prior.reseed(self.gen_newseed())
#     for m in self.models:
#         m.reseed(self.gen_newseed())
#     if self.proposal is not None:
#         self.proposal.reseed(self.gen_newseed())


class Worker(mp.Process):
    def __init__(
        self,
        n: int,
        queue: mp.Queue,
        pipe: mp.Pipe,
        simulator: Callable,
        seed: int = None,
    ):
        """
        Args:
            n: Worker index
            queue: Queue
            pipe: Pipe
            simulator: Simulator to run on worker
            seed: Seed
        """
        super().__init__()
        self.n = n
        self.queue = queue
        self.pipe = pipe
        self.model = simulator
        self.rng = np.random.RandomState(seed=seed)
        # TODO seeding not implemented yet: https://github.com/mackelab/sbi/issues/166

    def update(self, i):
        self.queue.put(i)

    def run(self):
        """
        Run simulations on worker.

        Updates simulation results in queue
        """
        logging.info(f"Worker {self.n}: Starting worker")
        while True:
            try:
                logging.info(f"Worker {self.n}: Listening")
                theta_worker_batch = self.pipe.recv()
            except EOFError:
                logging.info(f"Worker {self.n}: Leaving")
                break
            if len(theta_worker_batch) == 0:
                logging.info(f"Worker {self.n}: Skipping")
                self.pipe.send(([], []))
                continue

            # run forward model for all params, each n_reps times
            logging.info(
                f"Worker {self.n}: Received data of size {len(theta_worker_batch)}"
            )
            _, x = self.model(theta_worker_batch)

            logging.info(f"Worker {self.n}: Sending data")
            self.queue.put((x, theta_worker_batch))
            logging.info(f"Worker {self.n}: Done")
