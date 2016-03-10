from rlpy.Domains import Pinball
from rlpy.Agents import Q_Learning
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
#from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os
import numpy as np


"""
This class implements a path planner for rlpy
environments. It does so by rolling out
trajectories of policies
"""
class GoalPathPlanner:

	"""
	This class takes a domain as a parameter
	and learns a model to completion.
	"""
	def __init__(self, domain=None, representation=None, policy=None,steps=100000):
		

		if domain is None:
			return 

		opt = {}
		opt["domain"] = domain
		# Agent
		opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=238,
                       lambda_=0.9)
    
		opt["checks_per_policy"] = 5
		opt["max_steps"] = steps
		opt["num_policy_checks"] = 5
		experiment = Experiment(**opt)
		experiment.run()
		self.policy = opt["agent"].policy
		self.domain = domain

	def replace_policy(self, policy):
		self.policy = policy

	def replace_domain(self, domain):
		self.domain = domain

	"""
	Using the policy this generates a set
	of trajectories
	"""
	def generateTrajectories(self, N=10):
		demonstrations = []
		self.policy.turnOffExploration()
		for i in range(0, N):
			traj = []
			cur_state = self.domain.s0()[0]
			for j in range(0, self.domain.episodeCap):
				traj.append(cur_state)
				terminal = self.domain.isTerminal()

				if terminal:
					break

				a = self.policy.pi(cur_state, terminal,self.domain.possibleActions())
				total_results = self.domain.step(a)
				cur_state = total_results[1]
				
			demonstrations.append(traj)
		return demonstrations



