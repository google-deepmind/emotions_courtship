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

""""Config for play model."""

import ml_collections as config_dict


def get_config() ->  config_dict.ConfigDict:
  """Get default config for runs."""
  return config_dict.ConfigDict({
      "population": 500,
      "iterations": 100_000,
      "deaths": 0.01,
      "mutation": 0.04,
      "noise": "percept",  # "percept" or "action"
      "noisetype": 0.,
      "playcost": 0.,
      "deceitcost": 16.,
      "divorcecost": 1.,
      "allowplay": True,
      "allowdeceit": True,
      "allowdivorce": True,
      "emotions": True,
      "replicas": 14,
      "sample_freq": 500,
      "top_n": 25,
      "log_genotypes": False,
      "R": 3.,
      "T": 5.,
      "P": 0.,
      "S": -1.,
      "std_dev_initial_trait_value": 0.25,
      "initial_trait_value": 0.0,
      "sigmoid": "piece-wise",  # "piece-wise" or "smooth"
  })
