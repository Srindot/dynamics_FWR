import mujoco
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp

# 1. Load the model
model = mujoco.MjModel.from_xml_path("bat_wing.xml")
# Put model on GPU
mx_model = mjx.put_model(model)
# Create data structure
mx_data = mjx.make_data(model)

def aerodynamic_forces(data):
    # dx is the state (position, velocity)
    
    # NOTE: data.qvel is a flat array of all joint velocities.
    # You will need to map site/body IDs to these indices to get 
    # specific wing node velocities.
    
    # Example placeholder for calculating forces
    # In JAX/MJX, we usually create a vector of zeros and update specific indices
    calculated_forces = jnp.zeros(mx_model.nv) 
    
    # ... Your BET logic here ...
    
    return calculated_forces

# 2. Define the step function and JIT compile it
# The @jax.jit decorator compiles this function to XLA (GPU/TPU)
@jax.jit
def step_fn(m, d):
    # Calculate aero forces
    forces = aerodynamic_forces(d)
    
    # Inject forces into the simulation
    d = d.replace(qfrc_applied=forces)
    
    # Step the physics
    return mjx.step(m, d)

# 3. Run the loop
print("Compiling...")
mx_data = step_fn(mx_model, mx_data) # First call triggers compilation
print("Running simulation...")

for _ in range(1000):
    mx_data = step_fn(mx_model, mx_data)
    
print("Done.")