import numpy as np
import mujoco
import mujoco.viewer
import os
import time
import hydra
from omegaconf import DictConfig

# Set the directory for the simulation environment's source code
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class MuJoCoSim:
    """Main class for setting up and running the MuJoCo simulation."""

    model: mujoco.MjModel
    data: mujoco.MjData

    def __init__(self, cfg):
        """Class constructor to set up simulation configuration."""
        self.cfg = cfg  # Save environment configuration

        # Load the MuJoCo model
        xml_path = os.path.join(CURRENT_DIR, "asset", "xml", "robot.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = cfg.simulation.timestep  # Load timestep from configuration

        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        # Start the simulation viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Initialize control parameters (initialized as zeros)
        self.tau_limit = np.array(cfg.control.tau_limit, dtype=np.double)

        # Simulation parameters
        self.iter_ = 0

    def get_joint_state(self):
        """Retrieve the joint position, velocity, and torque states."""
        q_ = self.data.qpos.astype(np.double)  # Joint positions
        dq_ = self.data.qvel.astype(np.double)  # Joint velocities
        tau_ = self.data.qfrc_actuator.astype(np.double)  # Joint torques (actuated forces)
        return q_[7:], dq_[6:], tau_[6:]

    def hybrid_control(self, kp, kd, qDes, qdDes, ff):
        """Hybrid control combining Kp, Kd, qDes, qdDes, and feedforward torque."""
        current_positions, current_velocities, _ = self.get_joint_state()
        torques = kp * (qDes - current_positions) + kd * (qdDes - current_velocities) + ff
        torques = np.clip(torques, -self.tau_limit, self.tau_limit)
        self.data.ctrl = torques

    def fix_base(self):
        """Fix the base position and orientation."""
        self.data.qpos[:7] = np.array([0, 0, 1, 1, 0, 0, 0])
        self.data.qvel[:6] = np.zeros(6)

    def run(self, control_callback):
        """Main loop of simulation, now controlled via external callback.

        Parameters:
            control_callback (function): A user-defined function that takes the current joint state as input
            and returns the desired control action (qDes, qdDes, and ff).
        """
        while self.data.time < 1000.0 and self.viewer.is_running():
            step_start = time.time()

            # Fix base if necessary
            if self.cfg.simulation.fix_base:
                self.fix_base()

            # Get the current joint state (position, velocity, torque)
            joint_positions, joint_velocities, joint_torques = self.get_joint_state()

            # Call the user-defined control callback to get the next control action
            kp, kd, qDes, qdDes, ff = control_callback(joint_positions, joint_velocities, joint_torques)

            # Apply the hybrid control
            self.hybrid_control(kp, kd, qDes, qdDes, ff)

            mujoco.mj_step(self.model, self.data)

            if self.iter_ % 10 == 0:  # Sync at 50Hz
                self.viewer.sync()

            self.iter_ += 1
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


# Example control function
def user_controller(joint_positions, joint_velocities, joint_torques):
    """User-defined control logic."""
    # Example: target positions and zero velocities for hybrid control
    kp = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0], dtype=np.double)
    kd = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.double)
    qDes = np.array([-0.1, 0.2, 0.3, -0.1, -0.2, -0.3], dtype=np.double)
    qdDes = np.zeros_like(qDes)
    ff = np.zeros_like(qDes)

    return kp, kd, qDes, qdDes, ff


@hydra.main(
    version_base=None,
    config_name="pf_config",
    config_path=os.path.join(CURRENT_DIR, "cfg"),
)
def main(cfg: DictConfig) -> None:
    sim = MuJoCoSim(cfg)

    # Run simulation and pass in the user-defined controller
    sim.run(control_callback=user_controller)


if __name__ == "__main__":
    main()
