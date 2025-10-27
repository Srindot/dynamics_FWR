
import mujoco
import time

# Load the model
model = mujoco.MjModel.from_xml_path('model.xml')
data = mujoco.MjData(model)

print("Simulation starting... Press Ctrl+C to stop.")

# Run the simulation for 10 simulated seconds
while data.time < 10.0:
    mujoco.mj_step(model, data)
    
    # Print the simulation time
    print(f"Simulation time: {data.time:.2f} s", end='\r')
    
    # Sleep to make the output readable (not real-time)
    time.sleep(0.01)

print("\nSimulation finished.")
