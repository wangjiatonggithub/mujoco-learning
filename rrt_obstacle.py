import mujoco
import ompl.base as ob
import ompl.geometric as og
import time
import src.mujoco_viewer as mujoco_viewer
import src.pinocchio_kinematic as pinocchio_kinematic
import src.kdl_kinematic as kdl_kinematic
import src.key_listener as key_listener
import src.utils as utils
import numpy as np
from pynput import keyboard

key_states = {
    keyboard.Key.down: False
}

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, rendor_path, arm_path, init_qpos_max_attempts: int = 50, fixed_init_qpos: np.ndarray | None = None):
        super().__init__(rendor_path, 3, azimuth=-90, elevation=-30)
        self.arm_path = arm_path
        self.init_qpos_max_attempts = max(1, int(init_qpos_max_attempts))
        self.fixed_init_qpos = fixed_init_qpos
        
        self.obstacles_size = []
        self.obstacles_pos = []
        self.obstacles_rgba = []
        self.obstacles_type = []

        # self.obstacles_type.append("box")
        # self.obstacles_size.append([0.07, 0.07, 0.07])
        # self.obstacles_pos.append([0.25, 0.22, 0.5])
        # self.obstacles_rgba.append([0.4, 0.3, 0.3, 0.8])

        # obstacle_u_left
        self.obstacles_type.append("box")
        self.obstacles_size.append([0.010, 0.080, 0.300])
        self.obstacles_pos.append([-0.300, 0.300, 0.300])
        self.obstacles_rgba.append([0.300, 0.300, 0.300, 0.800])

        # obstacle_u_right
        self.obstacles_type.append("box")
        self.obstacles_size.append([0.010, 0.080, 0.300])
        self.obstacles_pos.append([0.900, 0.300, 0.300])
        self.obstacles_rgba.append([0.300, 0.300, 0.300, 0.800])

        # obstacle_u_base
        self.obstacles_type.append("box")
        self.obstacles_size.append([0.600, 0.010, 0.300])
        self.obstacles_pos.append([0.300, 0.370, 0.300])
        self.obstacles_rgba.append([0.300, 0.300, 0.300, 0.800])

        # self.obstacles_type.append("sphere")
        # self.obstacles_size.append([0.06])
        # self.obstacles_pos.append([0.3, 0.2, 0.5])
        # self.obstacles_rgba.append([0.3, 0.3, 0.3, 0.8])
        self.addObstacles(
            self.obstacles_pos,
            self.obstacles_type,
            self.obstacles_size,
            self.obstacles_rgba,
            out_xml_path="model/franka_emika_panda/rrt_obstacle.xml",
        )
        
        self.key_listener = key_listener.KeyListener(key_states)
        self.key_listener.start()
    
    def getBipolarJoints(self):
        self.initial_pos = self.model.key_qpos[0].copy()
        if self.fixed_init_qpos is not None:
            fixed = np.array(self.fixed_init_qpos, dtype=float).flatten()
            n = min(self.model.nq, fixed.shape[0])
            self.initial_pos[:n] = fixed[:n]
            for i in range(self.model.nq):
                self.data.qpos[i] = self.initial_pos[i]
            mujoco.mj_forward(self.model, self.data)
            if self.data.ncon != 0:
                print("Start pose in collision, abort.")
                self.start_dof = self.data.qpos[:9].copy()
                return False
        else:
            joint_ranges = self.model.jnt_range[:7]
            max_attempts = self.init_qpos_max_attempts
            found = False
            for _ in range(max_attempts):
                random_qpos = np.random.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
                self.initial_pos[:7] = random_qpos
                for i in range(self.model.nq):
                    self.data.qpos[i] = self.initial_pos[i]
                mujoco.mj_forward(self.model, self.data)
                if self.data.ncon == 0:
                    found = True
                    break
            if not found:
                for i in range(self.model.nq):
                    self.data.qpos[i] = self.initial_pos[i]
                mujoco.mj_forward(self.model, self.data)
        print("start dof", self.data.qpos[:self.model.nq])
        self.start_dof = self.data.qpos[:9].copy()

        euler = np.random.uniform(0, 2*np.pi, 3)
        tf = utils.transform2mat(self.goal_x, self.goal_y, self.goal_z, euler[0], euler[1], euler[2])
        self.solver = pinocchio_kinematic.Kinematics(self.ee_body_name)
        self.solver.buildFromMJCF(self.arm_path)
        self.dof, info = self.solver.ik(tf, current_arm_motor_q=self.start_dof)
        print("goal dof", self.dof)
        self.goal_dof = self.dof
        if len(self.goal_dof) < 9:
            self.goal_dof = np.concatenate((self.goal_dof, self.start_dof[7:]))
        return True
        
    def pathPlaning(self):
        state_space = ob.RealVectorStateSpace(self.model.nq)
        bounds = ob.RealVectorBounds(self.model.nq)
        for i in range(min(self.model.nq, self.model.jnt_range.shape[0])):
            bounds.setLow(i, self.model.jnt_range[i, 0])
            bounds.setHigh(i, self.model.jnt_range[i, 1])
        state_space.setBounds(bounds)
        si = ob.SpaceInformation(state_space)

        def is_state_valid(state):
            self.data.qpos[:7] = [state[i] for i in range(7)]
            mujoco.mj_step(self.model, self.data)
            return self.data.ncon == 0

        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
        si.setup()
        start = ob.State(state_space)
        goal = ob.State(state_space)
        for i in range(min(self.model.nq, self.model.jnt_range.shape[0])):
            start[i] = self.start_dof[i]
            goal[i] = self.goal_dof[i]

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        opt = ob.PathLengthOptimizationObjective(si)
        pdef.setOptimizationObjective(opt)
        planner = og.RRTConnect(si)
        self.planning_range = 0.01
        planner.setRange(self.planning_range)
        planner.setIntermediateStates(True)
        planner.setProblemDefinition(pdef)
        planner.setup()
        self.planning_timeout = 3.0
        solved = planner.solve(self.planning_timeout)
        self.path_states = []
        if solved:
            self.path = pdef.getSolutionPath()
            for i in range(self.path.getStateCount()):
                state = self.path.getState(i)
                state_values = [state[i] for i in range(self.model.nq)]
                self.path_states.append(state_values)
                # print(state_values)
        else:
            print("No solution found.")
        self.index = 0
        return solved

    def is_path_collision_free(self) -> bool:
        if not self.path_states:
            return False
        for state_values in self.path_states:
            self.data.qpos[:7] = state_values[:7]
            mujoco.mj_forward(self.model, self.data)
            if self.data.ncon != 0:
                return False
        return True
    
    def createTask(self):
        try_cnt = 20
        self.success = False
        for i in range(try_cnt):
            if self.getBipolarJoints() is False:
                return
            self.success = self.pathPlaning()
            if self.success and not self.is_path_collision_free():
                print("Path has collision, retry...", i)
                self.success = False
            if self.success:
                break
            print("Try again... cnt ", i)
    
    def runBefore(self):
        self.model.opt.timestep = 0.005
        self.ee_body_name = "ee_center_body"
        # self.goal_x = 0.4
        # self.goal_y = 0.4
        # self.goal_z = 0.5
        # 0.0 0.6 0.5
        self.goal_x = 0.0
        self.goal_y = 0.6
        self.goal_z = 0.5
        self.usr_geom_size = []
        self.usr_geom_pos = []
        self.usr_geom_rgba = []
        self.usr_geom_type = []
        self.usr_geom_pos.append([self.goal_x, self.goal_y, self.goal_z])
        self.usr_geom_type.append("sphere")
        self.usr_geom_size.append([0.02])
        self.usr_geom_rgba.append([0.1, 0.3, 0.3, 0.8])
        self.addVisuGeom(self.usr_geom_pos, self.usr_geom_type, self.usr_geom_size, self.usr_geom_rgba)
        self.createTask()

    def runFunc(self):
        if not self.success:
            return 
        if(len(self.path_states) == 0):
            return
        if self.index < len(self.path_states):
            self.data.qpos[:7] = self.path_states[self.index][:7]
            self.index += 1
        else:
            self.data.qpos[:7] = self.path_states[-1][:7]
            if key_states[keyboard.Key.down]:
                print("re create task")
                self.index = 0
                self.createTask()
        
if __name__ == "__main__":
    SCENE_XML_PATH = 'model/franka_emika_panda/scene_pos.xml'
    ARM_XML_PATH = 'model/franka_emika_panda/panda_pos.xml'
    USE_FIXED_INIT = True
    fixed_init_qpos = np.array([
        2.19268981e+00, -3.08604128e-01, -2.53317220e+00,
        -2.20413778e+00,  2.17344682e-01,  1.24643313e+00,
        1.46122689e-01,  3.43951343e-04,  3.43951343e-04,
    ])
    test = Test(
        SCENE_XML_PATH,
        ARM_XML_PATH,
        fixed_init_qpos=fixed_init_qpos if USE_FIXED_INIT else None,
    )
    test.run_loop()