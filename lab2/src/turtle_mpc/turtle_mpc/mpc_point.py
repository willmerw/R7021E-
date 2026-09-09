import do_mpc # Importing the MPC library

import casadi as ca
from casadi import vertcat, horzcat, DM # Importing specific functions from casadi for symbolic mathematics
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import math

class PointMPC(Node):
    def __init__(self):
        super().__init__('turtle_mpc_node')

        self.cmd_vel_publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.mpc_traj_publisher = self.create_publisher(Path, '/mpc_trajectory', 10)
        self.traj_publisher = self.create_publisher(Path, '/trajectory', 10)

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.traj = []

        self.model = defineTBotModel()
        self.model.setup()

        self.x_desired = 5.0
        self.y_desired = 0.0
        ts = 0.1
        N = 20 # Horizon
        self.mpc = defineTBotMPC(self.model, ts, N,self.x_desired, self.y_desired)
        self.mpc.settings.supress_ipopt_output = True
        self.mpc.setup()




    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        self.yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def timer_callback(self):



        p0 = np.array([self.x, self.y, self.yaw])

        d = np.linalg.norm(np.array([self.x_desired, self.y_desired]) - p0[:2])

        # Termination condition
        if d < 0.1:
            new_cmd = TwistStamped()
            new_cmd.twist.linear.x = 0.0
            new_cmd.twist.angular.z = 0.0
            self.cmd_vel_publisher.publish(new_cmd)
            return

        self.mpc.x0 = p0
        self.mpc.set_initial_guess()

        p = p0.copy()

        self.traj.append([self.x, self.y])
        traj_msg = Path()
        traj_msg.header.frame_id = 'odom'
        for x,y in self.traj:
            pose = PoseStamped()
            pose.header = traj_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            traj_msg.poses.append(pose)

        self.traj_publisher.publish(traj_msg)


        u = self.mpc.make_step(p)

        x_pred = self.mpc.data.prediction(('_x', 'x')).flatten()
        y_pred = self.mpc.data.prediction(('_x', 'y')).flatten()
        pred_trajectory = np.hstack([x_pred, y_pred])

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'

        for x, y in zip(x_pred, y_pred):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            path_msg.poses.append(pose)
        self.mpc_traj_publisher.publish(path_msg)

        new_cmd = TwistStamped()
        new_cmd.twist.linear.x = u[0][0]
        new_cmd.twist.angular.z = u[1][0]
        self.cmd_vel_publisher.publish(new_cmd)


def draw_eight(self, x, y, diameter):
    pts = []
    fid = 10
    for i in range(-180,180,fid):
        rad = math.radians(i)
        x_pos = (x + diameter) + diameter*math.cos(rad)
        y_pos = y + diameter*math.sin(rad)
        pts.append((x_pos,y_pos))
    for i in range(360,0,-fid):
        rad = math.radians(i)
        x_pos = (x - diameter) + diameter*math.cos(rad)
        y_pos = y + diameter*math.sin(rad)
        pts.append((x_pos,y_pos))
    return pts


def defineTBotModel():

  model_type = 'continuous'
  model = do_mpc.model.Model(model_type)

  # States
  x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
  y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
  th = model.set_variable(var_type='_x', var_name='th', shape=(1,1))

  # Inputs
  vx = model.set_variable(var_type='_u', var_name='vx')
  vt = model.set_variable(var_type='_u', var_name='vt')

  # Dynamics
  model.set_rhs('x', vx*ca.cos(th))
  model.set_rhs('y', vx*ca.sin(th))
  model.set_rhs('th', vt)

  return model

def defineTBotMPC(model, ts, N, x_desired, y_desired):
  mpc = do_mpc.controller.MPC(model)
  setup_mpc = {
      'n_horizon': N,
      't_step': ts,
      'n_robust': 1,
      'store_full_solution': True
  }
  mpc.set_param(**setup_mpc)

  mterm = (model.x['x']-x_desired)**2 + (model.x['y']-y_desired)**2
  lterm = (model.x['x']-x_desired)**2 + (model.x['y']-y_desired)**2
  mpc.set_objective(mterm=mterm, lterm=lterm)
  mpc.set_rterm(vx=1e-2, vt=1e-2)

  # State Bounds
  mpc.bounds['lower','_x','x'] = -10.0
  mpc.bounds['upper','_x','x'] =  10.0
  mpc.bounds['lower','_x','y'] = -10.0
  mpc.bounds['upper','_x','y'] =  10.0

  # Input Constraints
  mpc.bounds['lower','_u','vx'] = 0.0
  mpc.bounds['upper','_u','vx'] =  0.5
  mpc.bounds['lower','_u','vt'] = -0.8
  mpc.bounds['upper','_u','vt'] =  0.8

  # Nonlinear Constraint for Obstacle 1
  x_obs1 = 3.0
  y_obs1 = 0.0
  r_obs1 = 0.2
  mpc.set_nl_cons(
      'obs1',
      r_obs1**2 - ((model.x['x']-x_obs1)**2 + (model.x['y']-y_obs1)**2),
      ub=0.0,
      soft_constraint=False   # <-- no penalty, enforced strictly
  )

  # Nonlinear Constraint for Obstacle 2
  x_obs2 = 1.4
  y_obs2 = -0.3
  r_obs2 = 0.3
  mpc.set_nl_cons(
      'obs2',
      r_obs2**2 - ((model.x['x']-x_obs2)**2 + (model.x['y']-y_obs2)**2),
      ub=0.0,
      soft_constraint=False   # <-- no penalty, enforced strictly
  )

  return mpc

def main(): # Main Function
    rclpy.init()
    node = TurtleMPC()
    rclpy.spin(node) # Keep the node alive and runnings.
    node.destroy_node() # Destroy the node after completion (typically when exited).
    rclpy.shutdown() # Shutdown the ros client on interruption.


if __name__=='__main__': # Entry point
    main()