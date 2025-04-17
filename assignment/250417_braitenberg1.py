import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from IPython.display import Video

%matplotlib inline

dt = 0.05
T = 10
steps = int(T / dt)
wheel_base = 1.0

stimulus = np.array([0.0, 0.0])

sensor_offset = 0.6 # Distance from vehicle center to sensor
sensor_angle = np.pi / 4 #relative to vehicle forward direction

def sensor_A_value(sensor_pos, other_pos):
    d_source = np.linalg.norm(sensor_pos - stimulus)
    d_other  = np.linalg.norm(sensor_pos - other_pos)

    lobe1 = np.exp(-((d_source - 7)**2) / 2)
    lobe2 = np.exp(-((d_other - 7)**2) / 2)

    val = (lobe1 + lobe2) / 2
    return val


def sensor_B_value(sensor_pos, other_pos):
    d_source = np.linalg.norm(sensor_pos - stimulus)
    val = (np.exp(-((d_source - 12.0)**2) / 2))
    return val


def nonlinear_speed(I):
    return 25.0 * np.exp(-((I - 1.0)**2) / (2 * 0.5**2))

stateA = np.array([-2.0, 4, np.pi])
stateB = np.array([-2.0, 12.0, 0.0])

trajA = [stateA[:2].copy()]
trajB = [stateB[:2].copy()]

def draw_vehicle(ax, pos, theta, color):
    """
    Draws the vehicle as a rotated rectangle and its sensor beams as translucent triangles.
    """
    L = 1.5  # Vehicle body length
    W = 1.0  # Vehicle body width
    dx = L / 2
    dy = W / 2
    # Define rectangle corners in vehicle-local coordinates.
    corners = np.array([
        [ dx,  dy],
        [-dx,  dy],
        [-dx, -dy],
        [ dx, -dy]
    ])
    # Rotate and translate.
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    rotated = corners @ R.T + pos
    poly = patches.Polygon(rotated, closed=True, color=color, ec='k')
    ax.add_patch(poly)
    
    # Sensor beams (drawn as triangles)
    fov = np.pi / 6         # Sensor field-of-view
    sensor_range = 1.2      # Length of sensor beam

    # Left sensor:
    left_center = pos + sensor_offset * np.array([np.cos(theta + sensor_angle),
                                                  np.sin(theta + sensor_angle)])
    left_pt1 = left_center + sensor_range * np.array([np.cos(theta + sensor_angle + fov/2),
                                                       np.sin(theta + sensor_angle + fov/2)])
    left_pt2 = left_center + sensor_range * np.array([np.cos(theta + sensor_angle - fov/2),
                                                       np.sin(theta + sensor_angle - fov/2)])
    left_tri = patches.Polygon(np.array([left_center, left_pt1, left_pt2]),
                               color=[1, 0.6, 0.6], alpha=0.4, ec='none')
    ax.add_patch(left_tri)
    
    # Right sensor:
    right_center = pos + sensor_offset * np.array([np.cos(theta - sensor_angle),
                                                    np.sin(theta - sensor_angle)])
    right_pt1 = right_center + sensor_range * np.array([np.cos(theta - sensor_angle + fov/2),
                                                         np.sin(theta - sensor_angle + fov/2)])
    right_pt2 = right_center + sensor_range * np.array([np.cos(theta - sensor_angle - fov/2),
                                                         np.sin(theta - sensor_angle - fov/2)])
    right_tri = patches.Polygon(np.array([right_center, right_pt1, right_pt2]),
                                color=[1, 0.6, 0.6], alpha=0.4, ec='none')
    ax.add_patch(right_tri)

def update(frame):
    global stateA, stateB, trajA, trajB
    posA = stateA[:2]
    # Compute sensor positions for A:
    leftA = posA + sensor_offset * np.array([np.cos(stateA[2] + sensor_angle),
                                               np.sin(stateA[2] + sensor_angle)])
    rightA = posA + sensor_offset * np.array([np.cos(stateA[2] - sensor_angle),
                                                np.sin(stateA[2] - sensor_angle)])
    # Sensor readings using Vehicle A’s sensor function.
    S_A_left = sensor_A_value(leftA, stateB[:2])
    S_A_right = sensor_A_value(rightA, stateB[:2])
    # Ipsilateral wiring: left sensor → left motor, right sensor → right motor.
    v_left_A = nonlinear_speed(S_A_left)
    v_right_A = nonlinear_speed(S_A_right)
    vA = (v_left_A + v_right_A) / 2.0
    omegaA = (v_right_A - v_left_A) / wheel_base
    
    stateA[2] += omegaA * dt
    stateA[0] += vA * np.cos(stateA[2]) * dt
    stateA[1] += vA * np.sin(stateA[2]) * dt

    posB = stateB[:2]
    # Compute sensor positions for B:
    leftB = posB + sensor_offset * np.array([np.cos(stateB[2] + sensor_angle),
                                               np.sin(stateB[2] + sensor_angle)])
    rightB = posB + sensor_offset * np.array([np.cos(stateB[2] - sensor_angle),
                                                np.sin(stateB[2] - sensor_angle)])
    S_B_left = sensor_B_value(leftB, stateA[:2])
    S_B_right = sensor_B_value(rightB, stateA[:2])
    # Contralateral wiring: left sensor drives right motor and vice versa.
    v_left_B = nonlinear_speed(S_B_right)
    v_right_B = nonlinear_speed(S_B_left)
    vB = (v_left_B + v_right_B) / 2.0
    omegaB = (v_right_B - v_left_B) / wheel_base
    
    stateB[2] += omegaB * dt
    stateB[0] += vB * np.cos(stateB[2]) * dt
    stateB[1] += vB * np.sin(stateB[2]) * dt
    
    # Save trajectories.
    trajA.append(stateA[:2].copy())
    trajB.append(stateB[:2].copy())

    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    
    ax.plot([stimulus[0]], [stimulus[1]], "yo", markersize=15)
    
    trajA_np = np.array(trajA)
    trajB_np = np.array(trajB)
    ax.plot(trajA_np[:, 0], trajA_np[:, 1], "b-", lw=1, label="Vehicle A (8-shaped)")
    ax.plot(trajB_np[:, 0], trajB_np[:, 1], "r-", lw=1, label="Vehicle B (Circular)")
    
    # Draw the vehicles with sensor beams.
    draw_vehicle(ax, stateA[:2], stateA[2], "blue")
    draw_vehicle(ax, stateB[:2], stateB[2], "red")
    
    ax.legend(loc="upper right")
    ax.set_title("Braitenberg Vehicles: Circular (Blue) vs. Figure‑8 (Red)")

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
plt.show()

ani.save("braitenberg_vehicles_result.mp4", writer="ffmpeg", fps=30)
Video("braitenberg_vehicles_result.mp4", embed=True)
