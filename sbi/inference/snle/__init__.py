# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

raise ImportError(
    "You imported a file from `sbi.inference.snle`. However, as of sbi v0.23.0, this "
    "import is no longer supported. Instead, you have to import methods from "
    "`sbi.inference.trainers.nle` (notice the renaming from snle to nle). For example: "
    "`from sbi.inference.trainers.nle import SNLE_A` "
    "Please create an issue if you experience unexpected behavior."
)
