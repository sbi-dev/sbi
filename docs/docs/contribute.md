## Issues

Open [issues on GitHub](https://github.com/mackelab/sbi/issues) for any problems you encounter. 


## Pull requests

In general, we use pull requests to make changes to `sbi`.


## Code style

For docstrings and comments, we use [Google Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Code needs to pass through the following tools, which are installed along with `sbi`:

**[black](https://github.com/psf/black)**: Automatic code formatting for Python. You can run black manually from the console using `black .` in the top directory of the repository, which will format all files. 

**[isort](https://github.com/timothycrosley/isort)**: Used to consistently order imports. You can run isort manually from the console using `isort -y` in the top directory.


## Documentation

The documentation is entirely written in markdown ([basic markdown guide](https://guides.github.com/features/mastering-markdown/)). It would be of great help, if you fixed mistakes, and edited where things are unclear. After a PR with documentation changes has been merged, the online documentation will be updated automatically in a couple of minutes. If you want to test a local build of the documentation, take a look at `docs/README.md`.
