import numpy as np
from numpy import linalg as LA
import pybullet as p
import functools
import inspect
import copy
import warnings
warnings.filterwarnings('ignore')

ENT_COLOR_LIST = [
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 1, 1]
]

def dist(pos1, pos2):
    """get distance between two points"""
    return LA.norm(np.array(pos1) - np.array(pos2))


def list2color_txt(rgba):
    colors = np.array([[255/255, 87/255, 34/255, 1.0], [217/255, 191/255, 119/255, 1.0], [149/255, 56/255, 158/255, 1.0], [148/255, 252/255, 19/255, 1.0]])
    color_idx = np.argmin(np.linalg.norm(np.array(rgba) - colors, axis=1)) 
    return ['red', 'yellow', 'purple', 'green'][color_idx]    


def one_hot2shape(oh):
    return ['cube', 'cone', 'sphere', 'bowl'][np.argmax(oh)]


def tdw2pb(pos):
    return [pos[0], pos[2], pos[1]]

def tdw2pb_quat(pos):
    return [pos[0], pos[2], pos[1], pos[3]]

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return [qx, qy, qz, qw]


def quaternion_to_euler_angle_vectorized2(w, x, y, z):
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))
    return {"x":X, "y":Y, "z":Z}    

class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""
    def __init__(self, connection_mode=p.DIRECT, options=""):
        """Create a simulation and connect to it."""
        self._client = p.connect(p.SHARED_MEMORY)
        if (self._client < 0):
            print("options=", options)
            self._client = p.connect(connection_mode, options=options)
        self._shapes = {}


    def __del__(self):
        """Clean up connection if not already done."""
        try:
            p.disconnect(physicsClientId=self._client)
        except p.error:
            pass


    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(p, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute


class PyAgentEnv:

    def __init__(self, min_forces=None, max_forces=None, cond_text=None, mode=None):
        """min_forces, max_forces from inverse model training data for cliiping action in step
           cond_text, mode for calculating done"""
        self._p = BulletClient()
        self.min_forces = min_forces
        self.max_forces = max_forces
        self.cond_text = cond_text
        self.mode = mode
        self._action_repeat = 5 #10


    def load_scene(self, init_s):
        """init_s - 52 dim
           create agent and objects all spheres"""
        
        self.init_s = copy.deepcopy(init_s)
        self.entities_pos = [tdw2pb(init_s[:3]), tdw2pb(init_s[24:27]), tdw2pb(init_s[38:41])] #agent, target1, target2
        self._build()
        self.num_entities = 3
        self.num_agents = 1
        self.positions = [None] * self.num_entities
        self.orientations = [None] * self.num_entities
        self.colors = [0] * self.num_entities
        self._entities = [None] * self.num_entities
        self.trajectories = [None] * self.num_entities
        self.vels = [None] * self.num_entities

        for entity_id in range(self.num_entities):
            pos = self.entities_pos[entity_id]
            if entity_id == 0: #agent
                self._entities[entity_id] = \
                    self._create_sphere(sphereRadius=0.1, #0.2,
                                        position=pos,
                                        orientation=[0, 0, 0],
                                        mass=1,
                                        color=ENT_COLOR_LIST[self.colors[entity_id]])
                    # self._create_cone(coneRadius=0.2,
                    #                 coneHeight=0.4,
                    #                     position=pos,
                    #                     orientation=[0, 0, 0],
                    #                     mass=1,
                    #                     color=ENT_COLOR_LIST[self.colors[entity_id]])                                                                                                            
            else: #targets
                self._entities[entity_id] = \
                    self._create_sphere_obj(sphereRadius=0.1, #0.2,
                                        position=pos,
                                        orientation=[0, 0, 0],
                                        mass=1,
                                        color=ENT_COLOR_LIST[self.colors[entity_id]])
            self.positions[entity_id], self.orientations[entity_id] = \
                self._p.getBasePositionAndOrientation(self._entities[entity_id])
            self.trajectories[entity_id] = [list(self.positions[entity_id])]
            self.vels[entity_id] = [self._p.getBaseVelocity(self._entities[entity_id])]  
        # print('load_scene', self.positions)
        

    def _build(self):
        """build environment"""
        self._p.resetSimulation()
        self._num_bullet_solver_iterations = 10
        self._time_step = 1.0 / 120.0
        self._p.setPhysicsEngineParameter(
            numSolverIterations=self._num_bullet_solver_iterations)
        self._p.setTimeStep(self._time_step)
        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self._p.setGravity(0, 0, -9.81)
        self._p.setRealTimeSimulation(0)
        """base plane"""
        colBoxId = self._p.createCollisionShape(p.GEOM_PLANE)
        visBoxId = self._p.createVisualShape(p.GEOM_PLANE)
        boxId = self._p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=colBoxId,
                                        baseVisualShapeIndex=visBoxId,
                                        basePosition=[0, 0, 0],
                                        baseOrientation=[0, 0, 0])
        self._p.changeDynamics(boxId, -1, lateralFriction=0.4, rollingFriction=0.4, restitution=0.)
        self._floor_ids = []        
        self._floor_base = 0.01
        self._floor_height = self._floor_base #* 100
        base_height = self._floor_height
        self.width = 15
        self.height = 15

        colBoxId = self._p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=[self.width, 
                                     self.height, 
                                     base_height])
        visBoxId = self._p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[self.width, 
                                     self.height, 
                                     base_height],
                        rgbaColor=[0.8, 0.8, 0.8, 1])
        boxId = self._p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=colBoxId,
                                        baseVisualShapeIndex=visBoxId,
                                        basePosition=[0,0,0],
                                        baseOrientation=[0,0,0])
        self._p.changeDynamics(boxId, -1, lateralFriction=0.4)
        self._floor_ids.append(boxId)


    def step(self, force):
        """force - x,y(height),z 
           convert to x,y,z(height)"""
        if len(force) == 3:
            force = [force[0], force[2], force[1]]
        elif len(force) == 2:
            mag = 28.8 #29 #TDW -> pybullet
            force = [mag*force[0], mag*force[1], 0]
        entity_id = 0 #agent
        self._p.applyExternalForce(self._entities[entity_id],
                                   -1,
                                   force, 
                                   self.positions[entity_id],
                                   p.WORLD_FRAME)
        self._p.resetBaseVelocity(self._entities[entity_id],
                                      angularVelocity=[0, 0, 0])
        for _ in range(self._action_repeat): 
            self._p.stepSimulation()

        for entity_id in range(3):
            self.positions[entity_id], self.orientations[entity_id] = \
                self._p.getBasePositionAndOrientation(self._entities[entity_id])

        reward = self._reward()
        curr_state = self._get_obs()
        done = self._terminal(curr_state)
        
        # return curr_state#, reward, done
        return curr_state, done


    def _get_obs(self,):
        """tdw format - dim 52"""        
        velocity, angular_velocity = self._p.getBaseVelocity(self._entities[0])
        curr_state = tdw2pb(self.positions[0]) + tdw2pb_quat(list(self.orientations[0])) + tdw2pb(list(velocity)) + tdw2pb(list(angular_velocity)) + list(self.init_s[13:24]) + \
                    tdw2pb(self.positions[1]) + list(self.init_s[27:38]) + \
                    tdw2pb(self.positions[2]) + list(self.init_s[41:52])
        curr_state[1], curr_state[4], curr_state[8], curr_state[11], curr_state[25], curr_state[39]  = 0, 0, 0, 0, 0, 0 
        return np.array(curr_state)


    def _reward(self,):
        return None


    def _terminal(self, state):
        THRES = 0.365
        # THRES = np.inf
        agent_pos_0 = self.init_s[:3]
        agent_pos_H = state[:3] #current pos
        #predict only agent pos
        target_1_pos = self.init_s[24:27]
        target_1_color = list2color_txt(self.init_s[30:34])
        target_1_shape = one_hot2shape(self.init_s[34:38])
        target_2_pos = self.init_s[38:41]
        target_2_color = list2color_txt(self.init_s[44:48])
        target_2_shape = one_hot2shape(self.init_s[48:52])
        ag_t1_0 = dist(agent_pos_0, target_1_pos)
        ag_t1_H = dist(agent_pos_H, target_1_pos)
        ag_t2_0 = dist(agent_pos_0, target_2_pos)
        ag_t2_H = dist(agent_pos_H, target_2_pos)
        #target color
        if 'red' in self.cond_text: chosen_color = 'red'
        elif 'yellow' in self.cond_text: chosen_color = 'yellow'    
        elif 'purple' in self.cond_text: chosen_color = 'purple'
        elif 'green' in self.cond_text: chosen_color = 'green'
        else: chosen_color = None
        #target shape
        if 'bowl' in self.cond_text: chosen_shape = 'bowl'
        elif 'cube' in self.cond_text: chosen_shape = 'cube'    
        elif 'cone' in self.cond_text: chosen_shape = 'cone'    
        elif 'sphere' in self.cond_text: chosen_shape = 'sphere' 
        else: chosen_shape = None 
        #target num
        #reached target
        if self.mode == 'test' or self.mode == 'test_new':
            if target_1_color == chosen_color and target_1_shape == chosen_shape:
                return True if (ag_t1_H<ag_t1_0 and ag_t1_H < THRES) else False
            elif target_2_color == chosen_color and target_2_shape == chosen_shape:
                return True if (ag_t2_H<ag_t2_0 and ag_t2_H < THRES) else False  
        elif self.mode == 'train' or self.mode == 'val':
            if target_1_color == chosen_color or target_1_shape == chosen_shape:
                return True if (ag_t1_H<ag_t1_0 and ag_t1_H < THRES) else False
            elif target_2_color == chosen_color or target_2_shape == chosen_shape:
                return True if (ag_t2_H<ag_t2_0 and ag_t2_H < THRES) else False
        # out of boundaries
        max_x, min_x, max_z, min_z = 1.66, -1.66, -3.257, -4.355
        if agent_pos_H[0] > max_x or agent_pos_H[0] < min_x or agent_pos_H[2] > max_z or agent_pos_H[2] < min_z:
            print('out of boundaries')
            return True
        return False


    def _create_sphere(self, sphereRadius, position, orientation, mass, color):
        """add sphere entities"""
        # print(sphereRadius, position)
        colSphereId = self._p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        visSphereId = self._p.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=color)
        sphereId = self._p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=colSphereId,
                                           baseVisualShapeIndex=visSphereId,
                                           basePosition=position,
                                     baseOrientation=orientation)
        self._p.changeDynamics(sphereId, 
                               -1, 
                               mass=mass, 
                               lateralFriction=0.0,
                               spinningFriction=10000000.0,
                               rollingFriction=10000000,
                               linearDamping=0)
        return sphereId


    def _create_sphere_obj(self, sphereRadius, position, orientation, mass, color):
        """add sphere entities"""
        # print(sphereRadius, position)
        colSphereId = self._p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        visSphereId = self._p.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=color)
        sphereId = self._p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=colSphereId,
                                           baseVisualShapeIndex=visSphereId,
                                           basePosition=position,
                                     baseOrientation=orientation)
        self._p.changeDynamics(sphereId, 
                               -1, 
                               rollingFriction=0.1, restitution=0)
        return sphereId     