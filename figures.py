import numpy as np
from gl import V3
from myLibrary import *
OPAQUE = 0
REFLECTIVE = 1
TRANSPARENT = 2

WHITE = (1, 1, 1)


class DirectionalLight(object):
    def __init__(self, direction=V3(0, -1, 0), intensity=1, color=WHITE):

        self.direction = normalize(direction)
        self.intensity = intensity
        self.color = color


class AmbientLight(object):
    def __init__(self, strength=0, color=WHITE):
        self.strength = strength
        self.color = color

    def getColor(self):
        return (self.strength * self.color[0],
                self.strength * self.color[1],
                self.strength * self.color[2])


class PointLight(object):
    # Luz con punto de origen que va en todas direcciones
    def __init__(self, position=V3(0, 0, 0), intensity=1, color=WHITE):
        self.position = position
        self.intensity = intensity
        self.color = color


class Material(object):
    def __init__(self, diffuse=WHITE, spec=1, ior=1, texture=None, matType=OPAQUE):
        self.diffuse = diffuse
        self.spec = spec
        self.ior = ior
        self.texture = texture
        self.matType = matType


class Intersect(object):
    def __init__(self, distance, point, normal, texCoords, sceneObject):
        self.distance = distance
        self.point = point
        self.normal = normal
        self.texCoords = texCoords
        self.sceneObject = sceneObject


class Sphere(object):
    def __init__(self, center, radius, material=Material()):
        self.center = center
        self.radius = radius
        self.material = material

    def ray_intersect(self, orig, dir):

        L = subtract(self.center, orig)
        l = getVectorMagnitude(L)

        tca = dotProduct(L, dir)

        d = (l**2 - tca**2)
        if d > self.radius ** 2:
            return None

        thc = (self.radius**2 - d) ** 0.5
        t0 = tca - thc
        t1 = tca + thc

        if t0 < 0:
            t0 = t1

        if t0 < 0:
            return None

        # P = O + t * D
        # re = []

        # a = list(dir)
        # b = list(orig)

        # re.append(b[0]+(t0*a[0]))
        # re.append(b[1]+(t0*a[1]))
        # re.append(b[2]+(t0*a[2]))

        # hit = re
        ab = list(dir)
        newMatrix = []
        for i in range(len(ab)):
            res = 0
            res += ab[i] * t0
            newMatrix.append(res)

        hit = add(list(orig), newMatrix)
        normal = subtract(hit, self.center)

        normal = normalize(normal)

        u = 1 - ((np.arctan2(normal[2], normal[0]) / (2 * pi())) + 0.5)
        v = np.arccos(-normal[1]) / pi()

        uvs = (u, v)

        return Intersect(distance=t0,
                         point=hit,
                         normal=normal,
                         texCoords=uvs,
                         sceneObject=self)


class Plane(object):
    def __init__(self, position, normal, material=Material()):
        self.position = position
        self.normal = normalize(normal)  # normal / np.linalg.norm(normal)
        self.material = material

    def ray_intersect(self, orig, dir):
        # t = (( planePos - origRayo) dot planeNormal) / (dirRayo dot planeNormal)
        denom = dotProduct(dir, self.normal)

        if abs(denom) > 0.0001:
            num = dotProduct(subtract(list(self.position),
                             list(orig)), list(self.normal))
            t = num / denom
            if t > 0:
                # P = O + t * D
                adir = list(dir)
                newMatrix = []
                for i in range(len(adir)):
                    res = 0
                    res += adir[i] * t
                    newMatrix.append(res)

                hit = add(list(orig), newMatrix)

                return Intersect(distance=t,
                                 point=hit,
                                 normal=self.normal,
                                 texCoords=None,
                                 sceneObject=self)

        return None


class AABB(object):
    # Axis Aligned Bounding Box
    def __init__(self, position, size, material=Material()):
        self.position = position
        self.size = size
        self.material = material
        self.planes = []

        self.boundsMin = [0, 0, 0]
        self.boundsMax = [0, 0, 0]

        halfSizeX = size[0] / 2
        halfSizeY = size[1] / 2
        halfSizeZ = size[2] / 2

        # Sides
        self.planes.append(
            Plane(np.add(position, V3(halfSizeX, 0, 0)), V3(1, 0, 0), material))
        self.planes.append(
            Plane(np.add(position, V3(-halfSizeX, 0, 0)), V3(-1, 0, 0), material))

        # Up and down
        self.planes.append(
            Plane(np.add(position, V3(0, halfSizeY, 0)), V3(0, 1, 0), material))
        self.planes.append(
            Plane(np.add(position, V3(0, -halfSizeY, 0)), V3(0, -1, 0), material))

        # Front and Back
        self.planes.append(
            Plane(np.add(position, V3(0, 0, halfSizeZ)), V3(0, 0, 1), material))
        self.planes.append(
            Plane(np.add(position, V3(0, 0, -halfSizeZ)), V3(0, 0, -1), material))

        # Bounds
        epsilon = 0.001
        for i in range(3):
            self.boundsMin[i] = self.position[i] - (epsilon + self.size[i]/2)
            self.boundsMax[i] = self.position[i] + (epsilon + self.size[i]/2)

    def ray_intersect(self, orig, dir):
        intersect = None
        t = float('inf')

        uvs = None

        for plane in self.planes:
            planeInter = plane.ray_intersect(orig, dir)
            if planeInter is not None:
                # Si estoy dentro de los bounds
                if planeInter.point[0] >= self.boundsMin[0] and planeInter.point[0] <= self.boundsMax[0]:
                    if planeInter.point[1] >= self.boundsMin[1] and planeInter.point[1] <= self.boundsMax[1]:
                        if planeInter.point[2] >= self.boundsMin[2] and planeInter.point[2] <= self.boundsMax[2]:
                            # Si soy el plano mas cercano
                            if planeInter.distance < t:
                                t = planeInter.distance
                                intersect = planeInter

                                u, v = 0, 0

                                if abs(plane.normal[0]) > 0:
                                    # mapear uvs para eje X, uso coordenadas en Y y Z.
                                    u = (
                                        planeInter.point[1] - self.boundsMin[1]) / (self.boundsMax[1] - self.boundsMin[1])
                                    v = (
                                        planeInter.point[2] - self.boundsMin[2]) / (self.boundsMax[2] - self.boundsMin[2])

                                elif abs(plane.normal[1]) > 0:
                                    # mapear uvs para eje Y, uso coordenadas en X y Z.
                                    u = (
                                        planeInter.point[0] - self.boundsMin[0]) / (self.boundsMax[0] - self.boundsMin[0])
                                    v = (
                                        planeInter.point[2] - self.boundsMin[2]) / (self.boundsMax[2] - self.boundsMin[2])

                                elif abs(plane.normal[2]) > 0:
                                    # mapear uvs para eje Z, uso coordenadas en X y Y.
                                    u = (
                                        planeInter.point[0] - self.boundsMin[0]) / (self.boundsMax[0] - self.boundsMin[0])
                                    v = (
                                        planeInter.point[1] - self.boundsMin[1]) / (self.boundsMax[1] - self.boundsMin[1])

                                uvs = (u, v)

        if intersect is None:
            return None

        return Intersect(distance=intersect.distance,
                         point=intersect.point,
                         normal=intersect.normal,
                         texCoords=uvs,
                         sceneObject=self)
