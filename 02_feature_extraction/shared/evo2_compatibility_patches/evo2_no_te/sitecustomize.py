"""Force Evo2 7B checkpoints to use their supported BF16 projection fallback."""

import os
import sys

if os.environ.get("EVO2_SRC"):          # a source checkout of the evo2 package
    sys.path.insert(0, os.environ["EVO2_SRC"])
import evo2.models

evo2.models.HAS_TE = False
