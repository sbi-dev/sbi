raise ImportError(
    "You imported a file from `sbi.inference.snre`. However, as of sbi v0.23.0, this "
    "import is no longer supported. Instead, you have to import methods from "
    "`sbi.inference.trainers.nre` (notice the renaming from snre to nre). For example: "
    "`from sbi.inference.trainers.nre import SNRE_B` "
    "Please create an issue if you experience unexpected behavior."
)
