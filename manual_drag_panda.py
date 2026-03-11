"""手动拖动机械臂并记录实时关节角度"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import csv

# 1. 加载你的机械臂 XML 模型 (请替换为你自己的 XML 文件路径)
xml_path = "model/franka_emika_panda/rrt_obstacle.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 专家提示：为了方便你手动拖拽，建议临时关闭重力，
# 否则你一松手机械臂就掉下去了，很难录制平滑轨迹。
model.opt.gravity[:] = [0, 0, 0] 

# 2. 设置固定初始关节角
fixed_init_qpos = np.array([
    2.19268981e+00, -3.08604128e-01, -2.53317220e+00,
    -2.20413778e+00,  2.17344682e-01,  1.24643313e+00,
    1.46122689e-01,  3.43951343e-04,  3.43951343e-04,
])
data.qpos[:fixed_init_qpos.shape[0]] = fixed_init_qpos
mujoco.mj_forward(model, data)

# 3. 定义数据存储结构
recorded_trajectory = []
recording_frequency = 10 # 记录频率：10 Hz (每秒记录10次)
last_record_time = time.time()

# 目标点（与 rrt_obstacle 一致）
goal_pos = np.array([0.0, 0.6, 0.5], dtype=float)
goal_size = 0.02
goal_rgba = np.array([0.1, 0.3, 0.3, 0.8], dtype=float)

# 假设你要记录末端执行器的坐标，这里填入末端 body 或 site 的名字
# end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector_name')

print("仿真已启动！双击选中机械臂末端，按住 Ctrl + 鼠标右键 开始拖拽。")
print("按在终端按 Ctrl+C 结束录制并保存数据。")

try:
    # 3. 启动被动渲染器 (允许在主线程中运行控制逻辑)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            for i in range(model.nu): # 遍历每一个电机 (Actuator)
                # 找到这个电机驱动的是哪个关节 (Joint ID)
                jnt_id = model.actuator_trnid[i, 0]
                
                # 找到这个关节在状态数组 qpos 中的索引位置
                q_idx = model.jnt_qposadr[jnt_id]
                
                # 将该关节当前的实际物理位置，设为电机的控制目标
                data.ctrl[i] = data.qpos[q_idx]
            # 物理引擎步进
            mujoco.mj_step(model, data)
            
            # 绘制目标点
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[goal_size, 0.0, 0.0],
                pos=goal_pos,
                mat=np.eye(3).flatten(),
                rgba=goal_rgba,
            )

            # 同步画面
            viewer.sync()
            
            # 4. 实时记录逻辑
            current_time = time.time()
            if current_time - last_record_time >= (1.0 / recording_frequency):
                # 记录关节角度 qpos (对于 7-DoF 机械臂通常是 7 维向量)
                current_qpos = data.qpos.copy()
                
                # 记录末端空间位置 xpos (3维笛卡尔坐标 xyz)
                # current_ee_pos = data.xpos[end_effector_id].copy()
                
                # 将数据存入列表 (这里以只存 qpos 为例)
                recorded_trajectory.append(current_qpos)
                print(f"qpos: {current_qpos}")
                last_record_time = current_time
                
            # 控制仿真速度与真实时间一致
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("\n录制被手动中断，准备保存数据...")

# 5. 保存数据为 CSV 或 NPY 格式
recorded_trajectory = np.array(recorded_trajectory)
np.save("human_demonstration.npy", recorded_trajectory)
print(f"成功保存了 {len(recorded_trajectory)} 帧轨迹数据到 human_demonstration.npy！")