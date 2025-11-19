import mujoco
import mujoco.mjx as mjx
import jax.numpy as jnp

# 1. Load the model
model = mujoco.MjModel.from_xml_path("bat_wing.xml")
mx = mjx.put_model(model)

def aerodynamic_forces(dx):
    # dx is the state (position, velocity)
    
    # 1. Get velocity of every "node" in the wing skin
    # In MJX, you access this via dx.qvel (velocities)
    
    # 2. Calculate Angle of Attack (alpha) per node
    # This is your linear algebra: Cross product of velocity and wing chord vector
    
    # 3. Apply BET formulas
    # Lift = 0.5 * rho * V^2 * Cl * Area
    
    # 4. Return the force vector for each body
    calculated_forces = 0
    return calculated_forces

def step_fn(model, data):
    # Calculate aero forces
    forces = aerodynamic_forces(data)
    
    # Inject forces into the simulation
    data = data.replace(qfrc_applied=forces)
    
    # Step the physics
    return mjx.step(model, data)