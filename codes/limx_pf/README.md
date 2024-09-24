# LimX Biped robot

## Overview
This repository provides a simulation environment for the LimX Biped Robot using MuJoCo.

## Env settings

Install the essential packages:

```bash
pip install torch
pip install hydra-core
pip install mujoco
pip install omegaconf
```

You can run the simulation of a biped robot by the following command, then you will see that the robot move forwards in a constant speed of 1 m/s.

```bash
python biped_robot_sim.py
``` 

Adjusting Control Parameters:
You can modify the control parameters such as the proportional (Kp) and derivative (Kd) gains, desired positions (qDes), velocities (qdDes), and feedforward torques (ff) by editing the user_controller function inside biped_robot_sim.py.

Example user_controller function:
```
def user_controller(joint_positions, joint_velocities, joint_torques):
    # Define your control logic here
    kp = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0], dtype=np.double)
    kd = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.double)
    qDes = np.array([-0.1, 0.2, 0.3, -0.1, -0.2, -0.3], dtype=np.double)
    qdDes = np.zeros_like(qDes)
    ff = np.zeros_like(qDes)

    return kp, kd, qDes, qdDes, ff

```

Changing Simulation Parameters:
To modify the simulation behavior (such as timestep or whether the robot's base is fixed), you can edit the cfg/pf_config.yaml file. For example:
```
simulation:
  timestep: 0.001        # Timestep for the simulation
  fix_base: true         # Fix the robot's base in place

control:
  tau_limit: [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]  # Joint torque limits
```