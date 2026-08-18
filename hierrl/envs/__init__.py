"""Gymnasium registration for the tissue-retraction task."""
from gymnasium.envs.registration import register

from .tissue_retract import TissueRetractEnv

register(id="TissueRetract-v0", entry_point="hierrl.envs.tissue_retract:TissueRetractEnv",
         kwargs={"multidirectional": False}, max_episode_steps=500)
register(id="TissueRetract-v1", entry_point="hierrl.envs.tissue_retract:TissueRetractEnv",
         kwargs={"multidirectional": True}, max_episode_steps=500)

__all__ = ["TissueRetractEnv"]
