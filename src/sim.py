from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim
import numpy as np
import os


def simulate(file_name, sim_id):
    ideal = {
        'gyro_b': np.array([0.0, 0.0, 0.0]),
        'gyro_arw': np.array([0.0, 0.0, 0.0]),
        'gyro_b_stability': np.array([0.0, 0.0, 0.0]),
        'accel_b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
        'accel_vrw': np.array([0.0, 0.0, 0.0]),
        'accel_b_stability': np.array([0.0, 0.0, 0.0]),
    }

    imu = imu_model.IMU(accuracy=ideal, axis=6, gps=False)

    os.chdir('..')
    sim = ins_sim.Sim(
        fs=[100, 0, 0],
        motion_def='data/motion_def/' + file_name,
        ref_frame=0,  # NED reference frame
        imu=imu
    )

    sim.run(1)

    sim.results('data/sims/sim' + sim_id)
    sim.plot(['ref_accel', 'ref_gyro', 'ref_att_euler'])


if __name__ == '__main__':
    simulate('motion_def-rotation_only.csv', '1')
    pass
