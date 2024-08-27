raise ImportError(
    "You imported a file from `sbi.inference.snle`. However, as of sbi v0.23.0, this "
    "import is no longer supported. Instead, you have to import methods from "
    "`sbi.inference.trainers.nle` (notice the renaming from snle to nle). For example: "
    "`from sbi.inference.trainers.nle import SNLE_A` "
    "Please create an issue if you experience unexpected behavior."
)
