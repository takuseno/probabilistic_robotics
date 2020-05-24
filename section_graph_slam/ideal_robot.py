import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as patches
import matplotlib.animation as anm


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval
    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)

        elems = []

        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax)
        else:
            frames = int(self.time_span / self.time_interval) + 1
            interval = int(self.time_interval * 1000)
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                         frames=frames, interval=interval, repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        time_str = 't = %.2f[s]' % (self.time_interval * i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, 'one_step'):
                obj.one_step(self.time_interval)
            
class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color='black'):
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.sensor = sensor
        self.poses = [pose]

    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems += ax.plot([x, xn], [y, yn], color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color='black')

        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, 'draw'):
            self.agent.draw(ax, elems)

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu * math.cos(t0), nu * math.sin(t0), omega]) * time
        else:
            return pose + np.array([nu / omega * (math.sin(t0 + omega * time) - math.sin(t0)),
                                    nu / omega * (-math.cos(t0 + omega * time) + math.cos(t0)),
                                    omega * time])

class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None
        
    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker='*', label='landmarks', color='orange')
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], 'id:' + str(self.id), fontsize=10))
        
class Map:
    def __init__(self):
        self.landmarks = []
        
    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)
        
    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)
            
class IdealCamera:
    def __init__(self,
                         env_map,
                         distance_range=(0.5, 6.0),
                         direction_range=(-math.pi / 3, math.pi / 3)):
        self.map = env_map
        self.distance_range = distance_range
        self.direction_range = direction_range
        self.lastdata = []
        
    def visible(self, polarpos):
        if polarpos is None:
            return False
        return self.distance_range [0] <= polarpos[0] \
            and polarpos[0] <= self.distance_range[1] \
            and self.direction_range[0] <= polarpos[1] \
            and polarpos[1] <= self.direction_range[1]
    
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))
        self.lastdata = observed
        return observed

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color='pink')

    @classmethod
    def observation_function(self, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi:
            phi -= 2 * np.pi
        while phi < -np.pi:
            phi += 2 * np.pi
        return np.array([np.hypot(*diff), phi]).T
