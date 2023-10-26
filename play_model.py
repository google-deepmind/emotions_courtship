# Copyright 2023 The Authors.
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

"""Model for play in cooperation.

Agents interact initially at random. After interacting they may form a
relationship. The relationship has variables attachment,
cooperativeness and stakes.
- attach determines whether the relationship ends
- coop determines the likelihood of defection
- stakes determine investment in each interaction
Each of these variables changes according to agent traits depending on the
outcome of the previous interaction (R, T, S, P)
"""

import collections
import random

from absl import logging
import numpy as np
import tree

_REWARD_OUTCOME = "R"
_SUCKER_OUTCOME = "S"
_TEMPTATION_OUTCOME = "T"
_PUNISHMENT_OUTCOME = "P"


def _sigmoid(x):
  return 1 / (1 + np.exp(-5 * x))


def actual_run(params):
  """Actually run the experiment with the given parameters."""

  def clip(i, min_: float = -1, max_: float = 1):
    if i > max_:
      return max_
    if i < min_:
      return min_
    return i

  class Agent:
    """Agent class."""

    def __init__(
        self,
        discrete_buckets: int = 4,
        min_geno: float = -1,
        max_geno: float = 1,
    ):
      self._discrete_buckets = discrete_buckets
      self._min_geno = min_geno
      self._max_geno = max_geno
      self.partner = None
      self.geno = {
          "attach": {
              "BASE": 0,
              "VOL": 0,
              _REWARD_OUTCOME: 0,
              _TEMPTATION_OUTCOME: 0,
              _PUNISHMENT_OUTCOME: 0,
              _SUCKER_OUTCOME: 0,
          },
          "cooper": {
              "BASE": 0,
              "VOL": 0,
              _REWARD_OUTCOME: 0,
              _TEMPTATION_OUTCOME: 0,
              _PUNISHMENT_OUTCOME: 0,
              _SUCKER_OUTCOME: 0,
          },
          "deceit": {
              "BASE": 0,
              "VOL": 0,
              _REWARD_OUTCOME: 0,
              _TEMPTATION_OUTCOME: 0,
              _PUNISHMENT_OUTCOME: 0,
              _SUCKER_OUTCOME: 0,
          },
          "play": {
              "BASE": 0,
              "VOL": 0,
              _REWARD_OUTCOME: 0,
              _TEMPTATION_OUTCOME: 0,
              _PUNISHMENT_OUTCOME: 0,
              _SUCKER_OUTCOME: 0,
          },
      }
      self.phen = {"attach": 0, "cooper": 0, "deceit": 0, "play": 0}
      self.birth()

    def birth(self, parent=None):
      """New agent, either as initial pop (None parent) or from parent."""
      self.payoff = 0
      for t in self.geno:
        for r in self.geno[t]:
          if parent is None:
            self.geno[t][r] = self._calc_genotype(
                np.random.normal(params["initial_trait_value"],
                                 params["std_dev_initial_trait_value"])
            )
          else:
            self.geno[t][r] = self._calc_genotype(parent.geno[t][r])
        self.phen[t] = self.geno[t]["BASE"]
      if self.partner is not None:
        self.partner.end_relationship()
        self.partner = None

    def _calc_genotype(self, p_trait):
      return clip(
          np.random.normal(p_trait, params["mutation"]),
          self._min_geno,
          self._max_geno,
      )

    def start_relationship(self, p):
      self.partner = p

    def end_relationship(self):
      self.partner = None
      for t in self.geno:
        self.phen[t] = self.geno[t]["BASE"]

    def update_affect(self, other, result):
      """Update the affective state."""
      if (
          params["allowdeceit"]
          and (result == _SUCKER_OUTCOME or result == _PUNISHMENT_OUTCOME)
          and other.decide("deceit")
      ):
        other.payoff -= params["deceitcost"]
        if result == _SUCKER_OUTCOME:
          result = _REWARD_OUTCOME
        elif result == _PUNISHMENT_OUTCOME:
          result = _TEMPTATION_OUTCOME
      if (
          params["noisetype"] == "percept"
          and random.uniform(0, 1) < params["noise"]
      ):
        if result == _REWARD_OUTCOME:
          result = _SUCKER_OUTCOME
        elif result == _TEMPTATION_OUTCOME:
          result = _PUNISHMENT_OUTCOME
        elif result == _PUNISHMENT_OUTCOME:
          result = _TEMPTATION_OUTCOME
        elif result == _SUCKER_OUTCOME:
          result = _REWARD_OUTCOME
      if params["emotions"]:
        for t in self.geno:
          self.phen[t] += self.phen[t] + 2 * self.geno[t][result]
          self.phen[t] = clip(
              self.phen[t] + (self.geno[t]["BASE"] - self.phen[t]) *
              abs(self.geno[t]["VOL"])
          )
      else:
        for t in self.geno:
          self.phen[t] = self.geno[t][result]

    def update_relationship(self, result):
      self.payoff += params[result]
      if self.partner is not None:
        self.update_affect(self.partner, result)
        if params["allowdivorce"] and not self.decide("attach"):
          self.partner.payoff -= params["divorcecost"]
          self.partner.end_relationship()
          self.payoff -= params["divorcecost"]
          self.end_relationship()

    def decide(self, choice):
      if params["sigmoid"] == "piece-wise":
        if random.uniform(-0.5, 0.5) < self.phen[choice]:
          return True
        return False
      elif params["sigmoid"] == "smooth":
        return random.uniform(0, 1) < _sigmoid(self.phen[choice])

    def discretize(self):
      """Get a string representation of the agent as a discretized genotype."""
      discrete = tree.map_structure(
          lambda x: int(  # pylint: disable=g-long-lambda
              (x - self._min_geno) / (self._max_geno - self._min_geno)
              * self._discrete_buckets),
          self.geno,
      )
      # Capture special case of volatility.
      for key, value in discrete.items():
        value["VOL"] = int(
            2 * abs(self.geno[key]["VOL"])
            / (self._max_geno - self._min_geno)
            * self._discrete_buckets
        )
      attach = ["a"]
      cooper = ["c"]
      deceit = ["d"]
      play = [_PUNISHMENT_OUTCOME]
      order = (
          "BASE",
          "VOL",
          _REWARD_OUTCOME,
          _TEMPTATION_OUTCOME,
          _PUNISHMENT_OUTCOME,
          _SUCKER_OUTCOME,
      )
      for y in order:
        attach.append(str(discrete["attach"][y]))
        cooper.append(str(discrete["cooper"][y]))
        deceit.append(str(discrete["deceit"][y]))
        play.append(str(discrete["play"][y]))
      return "".join(attach + cooper + deceit + play)

  def run_replicas(params, stat_names):
    """Runs a number of replicas with the given params."""
    to_run = params["replicas"]
    for i in range(to_run):
      nsamples = int(params["iterations"] / params["sample_freq"])
      # run one replica
      stats, histo_stats, agents = run(params, stat_names, nsamples)
      res = {"Replica": i}  # record data
      for key in stats:
        res[key] = sum(stats[key]) / nsamples
      for key in histo_stats:
        res[key] = histo_stats[key]
      results = dict(**params, **res)
      if params["log_genotypes"]:
        results["agents"] = [a.geno for a in agents]
      logging.info("Results: %s", results)

  def run(params, stat_names, nsamples):
    agents = []  # all agents
    pool = []  # unpaired agents
    agents_used = []  # agents who have already played
    pool_used = []  # unpaired agents who have already played
    fitness = []
    for _ in range(params["population"]):  # make agents
      a = Agent()
      agents.append(a)
      pool.append(a)
    histo = collections.defaultdict(int)
    stats = {}
    for s in stat_names:
      stats[s] = np.zeros(nsamples)
    for t in range(params["iterations"]):
      if t % params["sample_freq"] == 0:
        sample = int(t / params["sample_freq"])
      else:
        sample = None
      while len(agents) > 1:
        pair_interact(
            params, sample, agents, pool, agents_used, pool_used, stats
        )  # choose pairs from agent list to play a PD round
      agents = agents_used.copy()  # return all agents to the agent list
      pool = pool_used.copy()  # unpartnered agents go in the pool list
      agents_used.clear()
      pool_used.clear()
      for a in agents:
        fitness.append(
            a.payoff
        )  # fitness depends on agent's total lifetime payoff
      minfit = min(fitness)
      fitness = [f - minfit for f in fitness]
      if sum(fitness) != 0:
        parents = random.choices(
            agents,
            weights=fitness,
            k=int(params["deaths"] * params["population"]),
        )  # choose parents with prob. based on fitness
      else:
        parents = random.choices(
            agents, k=int(params["deaths"] * params["population"])
        )
      fitness.clear()
      for p in parents:  # death & reproduction
        a = random.choice(agents)  # choose a random agent to die
        if a not in pool:  # end its relationship
          pool.append(a)
          pool.append(a.partner)
        a.birth(p)  # replace 'a' with a mutated copy of 'p'

      if sample is not None:  # record data
        snap_histo = collections.defaultdict(int)
        for a in agents:
          strategy = a.discretize()
          histo[strategy] += 1
          snap_histo[strategy] += 1
        print(
            len(snap_histo),
            sorted(snap_histo.items(), key=lambda x: x[1], reverse=True)[
                : params["top_n"]
            ],
        )  # top n modes
        interacted = (
            stats["p_R"][sample]
            + stats["p_T"][sample]
            + stats["p_P"][sample]
            + stats["p_S"][sample]
        )
        if interacted > 0:
          for s in [
              "p_R",
              "p_T",
              "p_P",
              "p_S",
          ]:
            stats[s][sample] /= interacted
        stats["unattached"][sample] = len(pool) / params["population"]

    modes = sorted(list(histo.items()), key=lambda x: x[1], reverse=True)[
        : params["top_n"]
    ]  # top n modes
    histo_stats = {}
    for i, mode in enumerate(modes):
      histo_stats[f"strat{i}"] = mode[0]
      histo_stats[f"prev{i}"] = mode[1]
    return stats, histo_stats, agents

  def pair_interact(
      params, sample, agents, pool, agents_used, pool_used, stats
  ):
    a = choose_and_remove(agents)
    if a.partner is None:
      if sample is not None:
        stats["new_relationships"][sample] += 1
      pool.remove(a)
      b = choose_and_remove(pool)
      stay = True
      if params["allowplay"]:
        counter = 0
        play_max = 20
        while (
            (a.decide("play") or b.decide("play"))
            and stay
            and counter < play_max
        ):
          stay = play_pretend(a, b)
          counter += 1
        if sample is not None:
          stats["play_count"][sample] += counter
          stats["play_count_squared"][sample] += counter * counter
      if stay:
        a.partner = b
        b.partner = a
        interact(a, b, sample, stats)
    else:
      b = a.partner
      interact(a, b, sample, stats)
    agents.remove(b)
    agents_used.append(a)
    agents_used.append(b)
    if a.partner is None:
      pool_used.append(a)
      pool_used.append(b)

  def interact(a, b, sample, stats):
    result_a, result_b = play(a, b)
    b.update_relationship(result_b)
    a.update_relationship(result_a)
    if sample is not None:
      count(sample, result_a, stats, a)
      count(sample, result_b, stats, b)

  def play(a, b):
    a_c = a.decide("cooper")
    b_c = b.decide("cooper")
    if params["noisetype"] == "action":
      if random.uniform(0, 1) < params["noise"]:
        a_c = not a_c
      if random.uniform(0, 1) < params["noise"]:
        b_c = not b_c
    if a_c and b_c:
      result_a = _REWARD_OUTCOME
      result_b = _REWARD_OUTCOME
    else:
      if a_c:
        result_b = _TEMPTATION_OUTCOME
        result_a = _SUCKER_OUTCOME
      elif b_c:
        result_a = _TEMPTATION_OUTCOME
        result_b = _SUCKER_OUTCOME
      else:
        result_b = _PUNISHMENT_OUTCOME
        result_a = _PUNISHMENT_OUTCOME
    return result_a, result_b

  def play_pretend(a, b):
    result_a, result_b = play(a, b)
    a.payoff -= params["playcost"]
    b.payoff -= params["playcost"]
    b.update_affect(a, result_b)
    a.update_affect(b, result_a)
    if not a.decide("attach") or not b.decide("attach"):
      a.end_relationship()
      b.end_relationship()
      return False
    return True

  def check_cooperated(result):
    if result == _REWARD_OUTCOME or result == _SUCKER_OUTCOME:
      return True
    else:
      return False

  def count(sample, result, stats, agent):
    if check_cooperated(result):
      stats["n_C"][sample] += 1
    stats["p_" + result][sample] += 1
    for p in agent.phen:
      stats[p][sample] += agent.phen[p]
    stats["payoff"][sample] += agent.payoff

  def choose_and_remove(source):
    a = random.choice(source)
    source.remove(a)
    return a

  stat_names = [
      "attach",
      "cooper",
      "deceit",
      "play",
      "payoff",
      "n_C",
      "p_R",
      "p_T",
      "p_P",
      "p_S",
      "unattached",
      "new_relationships",
      "play_count",
      "play_count_squared",
  ]
  for n in range(params["top_n"]):
    stat_names.append("strat"+str(n))
    stat_names.append("prev"+str(n))
  run_replicas(params, stat_names)
