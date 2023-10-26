# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary to run the Play model.

Can run locally with:

```sh
python3 run_play
```
"""

from typing import Sequence

from absl import app
from absl import logging
from ml_collections import config_flags

from . import play_model


_CONFIG = config_flags.DEFINE_config_file('config', default='config.py')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Running with config:\n%s', _CONFIG.value)
  play_model.actual_run(_CONFIG.value)


if __name__ == '__main__':
  app.run(main)
