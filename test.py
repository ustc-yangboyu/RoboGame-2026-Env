import mujoco
import mujoco.viewer
import time
import argparse

parser = argparse.ArgumentParser(description="RoboGame Environment")
parser.add_argument("--xml", type=str, default="env_without_color.xml", help="XML file path")
parser.add_argument("--sim_speed", type=float, default=1.0, help="Simulation speed factor")
args = parser.parse_args()

XML_PATH = args.xml
HEIGHT_ADD = 1.0
SIM_SPEED = args.sim_speed

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("正在抬高 free joint 物体...")
modified = []

for jnt_id in range(model.njnt):
    if model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
        qpos_adr = model.jnt_qposadr[jnt_id]
        
        old_z = data.qpos[qpos_adr + 2]
        data.qpos[qpos_adr + 2] += HEIGHT_ADD
        
        body_id = model.jnt_bodyid[jnt_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        modified.append(name)
        print(f"  {name}: z {old_z:.2f} -> {old_z + HEIGHT_ADD:.2f}")

mujoco.mj_forward(model, data)
print(f"\n共抬高 {len(modified)} 个物体，启动 Viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 8.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
    
    while viewer.is_running():
        step_start = time.time()
        
        mujoco.mj_step(model, data)        
        viewer.sync()
        
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep / SIM_SPEED:
            time.sleep(model.opt.timestep / SIM_SPEED - elapsed)

print("Viewer 已关闭")