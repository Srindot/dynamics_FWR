import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import jax
import jax.numpy as jnp
import time

# 1. Load Model (CPU)
model = mujoco.MjModel.from_xml_path("bat_wing.xml")
data = mujoco.MjData(model)

# 2. Move to GPU (MJX)
mx_model = mjx.put_model(model)
mx_data = mjx.make_data(model)

# ... [Your aerodynamic_forces function here] ...
def aerodynamic_forces(data):
    return jnp.zeros(mx_model.nv) # Placeholder

@jax.jit
def step_fn(m, d):
    forces = aerodynamic_forces(d)
    d = d.replace(qfrc_applied=forces)
    return mjx.step(m, d)

# 3. Launch Viewer (Passive Mode)
# We use "launch_passive" so we can control the loop ourselves
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # Compile the function first (warmup)
    print("Compiling JIT...")
    mx_data = step_fn(mx_model, mx_data)
    
    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()

        # --- GPU PHYSICS ---
        # Run multiple physics steps per render frame for speed
        # (Standard MoJoCo is 2ms step, 60Hz video = ~30 steps per frame)
        for _ in range(30):
            mx_data = step_fn(mx_model, mx_data)

        # --- SYNC TO CPU ---
        # This pulls the GPU data back to the CPU 'data' object
        mjx.get_data_into(mjx.get_data(mx_model, mx_data), model, data)

        # --- UPDATE GUI ---
        viewer.sync()

        # Slow down to match real-time (optional, otherwise it runs at 100000x speed)
        time_until_next_frame = model.opt.timestep - (time.time() - step_start)
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)