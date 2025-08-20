# Automous Drone Navigation using Reinforcement Learning

## Drone Hovering
Moves the drone to a spot right above it and have it stay in the area. The spot stays in the same location for each run.

![](Images/sac_drone_hover.gif)

## Random Point 
Moves the drone to a random spot and have it stay in the area. The spots location changes for each run.

![](Images/sac_drone_random.gif)

## Multiple Targets
Moves the drone to a random spot and have it stay in the area. The spots location changes after each contact with the location.

![](Images/sac_drone_targets.gif)

## Follow the Line
![](Images/sac_drone_follow.gif)

## Development Notes
- `VecNormalize`  can cause issue when used with a new environment along with a model that is already trained. It is good to save the VecNormalize if you want to use for further validation [Reinforcement Learning Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
