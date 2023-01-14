#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pygame
from math import sqrt, radians, cos, sin
import json
import os
import time

WIDTH, HEIGHT = 800, 800

class Vec:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, v):
        try:
            return Vec(self.x+v.x, self.y+v.y, self.z+v.z)
        except:
            print(self)
            print(v)
            raise

    def __sub__(self, v):
        return Vec(self.x-v.x, self.y-v.y, self.z-v.z)

    def __iadd__(self, v):
        self.x += v.x
        self.y += v.y
        self.z += v.z

    def __isub__(self, v):
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z

    def __mul__(self, o):
        if isinstance(o, Vec):
            return self.x*o.x + self.y*o.y + self.z*o.z

        return Vec(self.x*o, self.y*o, self.z*o)

    def __imul__(self, n):
        self.x *= n
        self.y *= n
        self.z *= n

    def __truediv__(self, n):
        return Vec(self.x/n, self.y/n, self.z/n)

    def cross(self, v):
        return Vec(
            self.y*v.z - self.z*v.y,
            self.z*v.x - self.x*v.z,
            self.x*v.y - self.y*v.x
        )

    def mag(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        mag = self.mag()
        if mag == 0:
            return Vec()

        return self/mag

    @property
    def _P(self):
        return self - Vec(0,0,1)*((self-Vec(0,0,-2)) * Vec(0,0,1))

    def apply_matrix(self, mat):
        x = self * Vec(*mat[0])
        y = self * Vec(*mat[1])
        z = self * Vec(*mat[2])
        #return Vec(x,y,z)
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vec({self.x}, {self.y}, {self.z})"

    def copy(self):
        return Vec(self.x, self.y, self.z)

class Scene:
    X = 5
    Z = 5
    light = Vec(-3,5,-3)*50
    textures = {}

    for f in os.listdir("./textures"):
        name = f.split(".")[0]
        textures[name] = pygame.image.load(os.path.join("./textures", f))

    print(textures)

    def __init__(self):
        self.anims = []
        self.blocks = []
        for z in range(self.Z):
            for x in range(self.X):
                col = [(204,194,194),(179,170,170)][(x+z)%2]
                self.blocks.append(Block(Vec(x,-1,z), col, default=True))
        self.step = 0
        self.rot_scene = Vec(0,45,0)
        self.rot_cam = Vec(-20,0,0)
        self.offset = Vec(WIDTH/2, 3*HEIGHT/4)
        self.zoom = 1
        self.last_mpos = None
        self.frame = 0
        self.index = 0
        self.anims = []
        self.face_count = 0

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            palette = data["palette"]
            for b in palette:
                if "model" in b:
                    with open(os.path.join("./models", b["model"]), "r") as mod_f:
                        b["model"] = self.load_model(mod_f.read())

            for b in data["blocks"]:
                pos = Vec(b["x"], b["y"], b["z"])
                block = palette[b["state"]]
                opacity = b["opacity"] if "opacity" in b else 1
                frame = b["frame"] if "frame" in b else 0
                index = b["index"] if "index" in b else 0
                self.blocks.append(Block(pos, frame=frame, index=index, opacity=0, **block))

            self.frames = data["frames"]

        self.update_anims()

    def render(self, surf):
        #surf.fill((255,255,255,255))

        #self.rot_scene.y += 0.5
        #self.rot_scene.y %= 360

        rot_mat_scene = self.get_rot_mat(self.rot_scene)
        rot_mat_cam = self.get_rot_mat(self.rot_cam)
        HX = self.X*50/2
        HZ = self.Z*50/2

        faces = []
        for block in self.blocks:
            if block.opacity != 0 and (block.frame <= self.frame or (block.frame == self.frame and block.index <= self.index)):
                mesh = block.get_mesh()
                mesh.scale(50)
                mesh.translate(Vec(-HX, 0, -HZ))
                mesh.apply_matrix(rot_mat_scene)
                mesh.apply_matrix(rot_mat_cam)
                #mesh.translate(Vec(HX, 0, HZ))
                for face in mesh.faces:
                    faces.append([
                        [mesh.pts[i] for i in face], list(block.col)+[block.opacity*255], block.texture
                    ])
                #mesh.draw(surf, block.col, self.light)

        faces = sorted(faces, key=lambda f: sum([pt.z for pt in f[0]])/len(f[0]), reverse=True)
        self.face_count = 0
        for pts, col, texture in faces:
            n = (pts[1]-pts[0]).cross(pts[-1]-pts[0])
            if n*Vec(0,0,1) > 0:
                self.face_count += 1
                c = pygame.Color(col)
                hsva = list(c.hsva)
                v = self.light - sum(pts, Vec())/len(pts)
                r = v.normalize()*n.normalize()
                hsva[2] *= 1-(r+1)/2
                c.hsva = hsva
                pts = [p._P*(-1)*self.zoom + self.offset for p in pts]
                pts = [(p.x, p.y) for p in pts]
                if c.a != 255:
                    s = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                    if texture is None:
                        pygame.draw.polygon(s, c, pts)
                    else:
                        self.draw_face(s, pts, self.textures[texture], c.a)
                    surf.blit(s, [0,0])
                else:
                    if texture is None:
                        pygame.draw.polygon(surf, c, pts)

                    else:
                        self.draw_face(surf, pts, self.textures[texture])
        #pygame.draw.line(surf, (255,255,255), [WIDTH/2, 0], [WIDTH/2, HEIGHT])
        #pygame.draw.line(surf, (255,255,255), [0, 3*HEIGHT/4], [WIDTH, 3*HEIGHT/4])
        #light = self.light._P
        #pygame.draw.circle(surf, (255,255,0), [WIDTH/2-light.x, 3*HEIGHT/4-light.y], 3)

    def get_rot_mat(self, rot):
        a, b, c = radians(rot.z), radians(rot.y), radians(rot.x)
        mat = [
            [cos(a)*cos(b), cos(a)*sin(b)*sin(c)-sin(a)*cos(c), cos(a)*sin(b)*cos(c)+sin(a)*sin(c)],
            [sin(a)*cos(b), sin(a)*sin(b)*sin(c)+cos(a)*cos(c), sin(a)*sin(b)*cos(c)-cos(a)*sin(c)],
            [-sin(b), cos(b)*sin(c), cos(b)*cos(c)]
        ]
        return mat

    def load_model(self, data):
        pts = []
        lines = []
        faces = []
        for line in data.split("\n"):
            if line == "": continue
            args = line.split(" ")
            if args[0] == "v":
                pts.append(Vec(float(args[1]), float(args[2]), float(args[3])))
            elif args[0] == "e":
                lines.append([int(args[1]), int(args[2])])
            elif args[0] == "f":
                faces.append([int(arg) for arg in args[1:]])

        return Mesh(pts, lines, faces)

    def process(self, events):
        mpos = pygame.mouse.get_pos()
        r = 1-mpos[1]/HEIGHT
        #self.rot_cam.x = (r*2-1)*90

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.offset = Vec(WIDTH/2, 3*HEIGHT/4)
                    self.rot_scene = Vec(0,45,0)
                    self.rot_cam = Vec(-20,0,0)

                elif event.key == pygame.K_SPACE:
                    self.frame += 1
                    self.index = 0
                    self.update_anims()

                elif event.key == pygame.K_UP:
                    self.index += 1
                    self.update_anims()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.zoom *= 1.1
                    self.zoom = round(self.zoom, 2)

                elif event.button == 5:
                    self.zoom *= 0.9
                    self.zoom = round(self.zoom, 2)

        if self.last_mpos:
            diff = Vec(*mpos)-Vec(*self.last_mpos)
            if pygame.mouse.get_pressed()[0]:
                self.rot_scene.y += diff.x/WIDTH*360
                self.rot_cam.x -= diff.y/HEIGHT*360

            elif pygame.mouse.get_pressed()[1]:
                self.offset = self.offset + diff

        self.last_mpos = mpos
        for anim in self.anims:
            anim.update()
        self.anims = list(filter(lambda a: not a.finished, self.anims))

    def update_anims(self):
        for b in self.blocks:
            if b.default: continue
            if b.frame == self.frame and b.index == self.index:
                anim = Animation(b, "opacity", 0, 1, 0.5)
                anim2 = Animation(b.pos, "y", b.pos.y+1, b.pos.y, 0.5)
                self.anims.append(anim)
                self.anims.append(anim2)
                anim.start()
                anim2.start()

            elif b.frame < self.frame or (b.frame == self.frame and b.index < self.index):
                b.opacity = 1

        if self.frame < len(self.frames):
            frame = self.frames[self.frame]
            anim = Animation(self.rot_scene, "y", self.rot_scene.y, frame["rot"], 2)
            self.anims.append(anim)
            anim.start()

    def draw_face(self, surf, face, texture, alpha=255):
        p0, p1, p2 = face[0], face[1], face[-1]
        p0, p1, p2 = Vec(*p0), Vec(*p1), Vec(*p2)
        vx, vy = (p1-p0), (p2-p0)

        for y in range(16):
            for x in range(16):
                col = texture.get_at([int((x+0.5)*texture.get_width()/16), int((y+0.5)*texture.get_height()/16)])
                if alpha != 255:
                    col.a = alpha
                a = p0 + vx*x/16 + vy*y/16
                b = p0 + vx*(x+1)/16 + vy*y/16
                c = p0 + vx*(x+1)/16 + vy*(y+1)/16
                d = p0 + vx*x/16 + vy*(y+1)/16
                pygame.draw.polygon(surf, col, [(a.x,a.y), (b.x,b.y), (c.x,c.y), (d.x,d.y)])

class Mesh:
    def __init__(self, pts, lines, faces):
        self.pts = pts
        self.lines = lines
        self.faces = faces

    def draw(self, surf, col, light):
        """for i1, i2 in self.lines:
            p1 = self.pts[i1]._P * (-1)
            p2 = self.pts[i2]._P * (-1)
            pygame.draw.line(surf, col, [p1.x+WIDTH/2, p1.y+3*HEIGHT/4], [p2.x+WIDTH/2, p2.y+3*HEIGHT/4])"""
        
        faces = sorted(self.faces, key=lambda face: sum([self.pts[i].z for i in face])/4)

        for i1, i2, i3, i4 in faces:
            p1 = self.pts[i1]
            p2 = self.pts[i2]
            p3 = self.pts[i3]
            p4 = self.pts[i4]
            n = (p2-p1).cross(p4-p1)
            if n*Vec(0,0,1) > 0:
                c = pygame.Color(col)
                hsva = list(c.hsva)
                v = light - (p1+p2+p3+p4)/4
                r = v.normalize()*n.normalize()
                hsva[2] *= (r+1)/2
                c.hsva = hsva
                pts = [p1, p2, p3, p4]
                pts = [p._P*(-1) + Vec(WIDTH/2, 3*HEIGHT/4) for p in pts]
                pts = [(p.x, p.y) for p in pts]
                pygame.draw.polygon(surf, c, pts)

    def translate(self, vec):
        for pt in self.pts:
            pt += vec

    def scale(self, n):
        for pt in self.pts:
            pt *= n

    def apply_matrix(self, mat):
        for pt in self.pts:
            pt.apply_matrix(mat)

    def copy(self):
        pts = [pt.copy() for pt in self.pts]
        lines = self.lines.copy()
        faces = self.faces.copy()
        return Mesh(pts, lines, faces)

class Block:
    def __init__(self, pos, col, opacity=1, texture=None, model=None, frame=0, index=0, default=False):
        self.pos = pos
        self.col = col
        self.opacity = opacity
        self.texture = None #texture
        self.model = model
        self.frame = frame
        self.index = index
        #self.start_at = None
        self.default = default

    def get_mesh(self):
        if self.model:
            model = self.model.copy()
            model.translate(self.pos)
            return model

        pts = [
            self.pos+Vec(0,0,0),
            self.pos+Vec(1,0,0),
            self.pos+Vec(1,1,0),
            self.pos+Vec(0,1,0),
            self.pos+Vec(0,0,1),
            self.pos+Vec(1,0,1),
            self.pos+Vec(1,1,1),
            self.pos+Vec(0,1,1)
        ]
        lines = [
            (0,1),
            (1,2),
            (2,3),
            (3,0),
            (4,5),
            (5,6),
            (6,7),
            (7,4),
            (0,4),
            (1,5),
            (2,6),
            (3,7)
        ]
        faces = [
            (0,1,2,3),
            (4,0,3,7),
            (5,4,7,6),
            (1,5,6,2),
            (3,2,6,7),
            (4,5,1,0)
        ]
        return Mesh(pts, lines, faces)

class Animation:
    def __init__(self, obj_, val, a, b, duration):
        self.obj = obj_
        self.val = val
        self.a = a
        self.b = b
        self.duration = duration
        self.start_at = None
        self.finished = False

    def start(self):
        self.start_at = time.time()
        self.finished = False

    def update(self):
        if self.start_at is None: return
        t = time.time()
        r = (t-self.start_at)/self.duration
        if r > 1:
            v = self.b
            self.finished = True
        else:
            r = 1 - (r-1)**2
            v = (self.b-self.a)*r + self.a

        setattr(self.obj, self.val, v)

if __name__ == "__main__":
    pygame.init()

    w = pygame.display.set_mode([WIDTH, HEIGHT])
    s = pygame.Surface([WIDTH,HEIGHT], pygame.SRCALPHA)

    clock = pygame.time.Clock()

    scene = Scene()
    scene.load("scenes/test.json")

    while True:
        pygame.display.set_caption(f"Ponder - {clock.get_fps():.2f}fps - {scene.face_count} faces drawn")

        events = pygame.event.get()
        scene.process(events)
        w.fill((0,0,0))
        scene.render(w)
        #w.blit(s, [0,0])

        pygame.display.flip()

        clock.tick(60)
