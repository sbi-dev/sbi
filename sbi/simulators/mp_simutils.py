import multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm
from typing import Callable, Tuple
import torch
from sys import platform
from torch import Tensor


def simulate_mp(
    simulator: Callable,
    theta: Tensor,
    num_workers: int = 4,
    worker_batch_size: int = 20,
    verbose: bool = False,
    show_progressbar: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Return parameters theta and simulated data x, simulated using multiprocessing on the
        passed simulator.

    Args:
        simulator: simulator function that will be executed on each worker
        theta: all simulation parameters
        num_workers: number of parallel workers to start
        worker_batch_size: Number of parameter sets that are processed per worker. A
            worker will receive this many parameter sets to simulate per call. Needs
            to be larger than sim_batch_size. A lower value creates overhead from
            starting workers frequently. A higher value leads to the simulation
            progressbar being updated less frequently (updates only happen after a
            worker is finished).
        verbose: whether to give information about the current state of the workers
        show_progressbar: whether to show a progressbar

    Returns: parameters theta and simulation outputs x. The order of theta is not
        necessarily the same as for the input variable theta, which is why we return it
        here.
    """

    queue, workers, pipes = start_workers(
        simulator=simulator, num_workers=num_workers, verbose=verbose
    )
    log("Started workers", verbose)
    final_x = []  # list of simulation outputs
    final_theta = []  # list of parameters
    minibatches = iterate_minibatches(theta, worker_batch_size)

    pbar = tqdm(
        total=len(theta),
        disable=not show_progressbar,
        desc="Running {0} simulations".format(len(theta)),
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
                log(
                    "Dispatching to worker (len = {})".format(len(theta_worker_batch)),
                    verbose,
                )
                pipe.send(theta_worker_batch)
                log("Done", verbose)

            num_remaining = len(active_list)
            while num_remaining > 0:
                log("Listening to worker", verbose)
                msg = queue.get()
                if isinstance(msg, int):
                    log("Received int", verbose)
                elif isinstance(msg, tuple):
                    log("Received results", verbose)
                    x, theta = msg
                    final_x.append(x)
                    final_theta.append(theta)
                    num_remaining -= 1
                    pbar.update(len(theta))
                else:
                    log(
                        "Warning: Received unknown message of type {}".format(
                            type(msg)
                        ),
                        verbose,
                    )

    stop_workers(workers, pipes, queue, verbose=verbose)

    # concatenate to get shape (num_samples, shape_of_single_x)
    x = torch.cat(final_x, dim=0)
    # concatenate to get shape (num_samples, shape_of_single_theta)
    theta = torch.cat(final_theta, dim=0)

    return theta, x


def start_workers(simulator: Callable, num_workers: int = 4, verbose: bool = False):
    """
    Start all workers.

    Args:
        simulator: simulator function that will be executed on each core
        num_workers: number of parallel workers to start
        verbose: whether to give information about the current state of the workers

    Returns: queue, workers, pipes
    """
    try:
        start_method = dict(linux="fork", darwin="fork", win32="spawn")
        try:
            mp.set_start_method(start_method[platform])
        except:
            raise KeyError("Platform not supported.")
    except RuntimeError:
        # try-except because start method can not be set multiple times within the same
        # session. So, if the user runs things outside of a __main__() function, e.g. in
        # a jupyter notebook, this would give an error from the second call of the
        # set_start_method() function on
        pass

    pipes = [mp.Pipe(duplex=True) for _ in range(num_workers)]
    queue = mp.Queue()
    workers = [
        Worker(
            i,
            queue,
            pipes[i][1],
            simulator,  # models[i] TODO https://github.com/mackelab/sbi/issues/166
            seed=None,  # self.rng.randint(low=0, high=2 ** 31), # TODO #166
            verbose=verbose,
        )
        for i in range(num_workers)
    ]
    pipes = [p[0] for p in pipes]

    log("Starting workers", verbose)
    for w in workers:
        w.start()

    log("Done", verbose)

    return queue, workers, pipes


def stop_workers(workers, pipes, queue, verbose=False):
    """
    Stop all workers.

    Args:
        workers: workers
        pipes: pipes
        queue: queue
        verbose: whether to give information about the current state of the workers
    """
    if workers is None:
        return

    log("Closing")
    for w, p in zip(workers, pipes):
        log("Closing pipe", verbose)
        p.close()

    for w in workers:
        log("Joining process", verbose)
        w.join(timeout=1)
        w.terminate()

    queue.close()


def iterate_minibatches(theta: Tensor, worker_batch_size: int = 20) -> Tensor:
    """
    Yields the thetas of shape [minibatch, shape_of_single_theta] from the entire
    parameter array theta

    Args:
        theta: all input parameters
        worker_batch_size: size of the minibatch to be passed to each worker

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


def log(msg, verbose: bool = False):
    """
    Print information about the current state of the worker.

    Args:
        msg: message to print
        verbose: whether to print the information or not.
    """
    if verbose:
        print("Parent: {}".format(msg))


class Worker(mp.Process):
    def __init__(
        self,
        n: int,
        queue: mp.Queue,
        pipe: mp.Pipe,
        simulator: Callable,
        seed: int = None,
        verbose: bool = False,
    ):
        """
        Args:
            n: worker index
            queue: queue
            pipe: pipe
            simulator: simulator to run on worker
            seed: seed
            verbose: whether to give information about the current state of the worker
        """
        super().__init__()
        self.n = n
        self.queue = queue
        self.verbose = verbose
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
        self.log("Starting worker")
        while True:
            try:
                self.log("Listening")
                theta_worker_batch = self.pipe.recv()
            except EOFError:
                self.log("Leaving")
                break
            if len(theta_worker_batch) == 0:
                self.log("Skipping")
                self.pipe.send(([], []))
                continue

            # run forward model for all params, each n_reps times
            self.log("Received data of size {}".format(len(theta_worker_batch)))
            _, x = self.model(theta_worker_batch)

            self.log("Sending data")
            self.queue.put((x, theta_worker_batch))
            self.log("Done")

    def log(self, msg):
        """
        Print information about the current state of the worker.

        Args:
            msg: message to print
        """
        if self.verbose:
            print("Worker {}: {}".format(self.n, msg))
