from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.core_types import EnvironmentSteps, EnvironmentEpisodes
from rl_coach.environments.gym_environment import GymEnvironmentParameters, GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

# from gym_tmu import env

#########
# Agent #
#########
agent_params = DQNAgentParameters()


###############
# Environment #
###############
env_params = GymVectorEnvironment(level='renforcement.gym_tmu.env:GymTMU')

###################
# Graph Scheduling #
####################
num_round_improve_steps = 100
num_round_heatup = 50
num_round_training = 200
num_round_evaluation = 100

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(num_round_improve_steps)
schedule_params.heatup_steps = EnvironmentSteps(num_round_heatup)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(num_round_training)
schedule_params.evaluation_steps = EnvironmentSteps(num_round_evaluation)

########################
# Create Graph Manager #
########################
# BasicRLGraphManager, créé un uniquement LevelManager entre l'Agent et l'Environnement
graph_manager = BasicRLGraphManager(agent_params=agent_params,
                                    env_params=env_params,
                                    schedule_params=schedule_params)
