import struct
from collections import namedtuple
from obj import _color
from myLibrary import *
from numpy import sin, cos, tan

from obj import Obj

OPAQUE = 0
REFLECTIVE = 1
TRANSPARENT = 2

MAX_RECURSION_DEPTH = 3

STEPS = 1

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z', 'w'])


def char(c):
    # 1 byte
    return struct.pack('=c', c.encode('ascii'))


def word(w):
    # 2 bytes
    return struct.pack('=h', w)


def dword(d):
    # 4 bytes
    return struct.pack('=l', d)


def baryCoords(A, B, C, P):
    # u es para A, v es para B, w es para C
    try:
        # PCB/ABC
        u = (((B.y - C.y) * (P.x - C.x) + (C.x - B.x) * (P.y - C.y)) /
             ((B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y)))

        # PCA/ABC
        v = (((C.y - A.y) * (P.x - C.x) + (A.x - C.x) * (P.y - C.y)) /
             ((B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y)))

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w


def reflectVector(normal, dirVector):
    # R = 2 * ( N . L) * N - L
    reflect = 2 * dotProduct(normal, dirVector)
    reflect = ml(normal, reflect)
    reflect = subtract(reflect, dirVector)
    reflect = normalize(reflect)
    return reflect


def refractVector(normal, dirVector, ior):
    # Snell's Law
    cosi = max(-1, min(1, dotProduct(dirVector, normal)))
    etai = 1
    etat = ior

    if cosi < 0:
        cosi = -cosi
    else:
        etai, etat = etat, etai
        normal = matriz_negada(list(normal))  # np.array(normal) * -1

    eta = etai/etat
    k = 1 - eta * eta * (1 - (cosi * cosi))

    if k < 0:  # Total Internal Reflection
        return None
    a = list(dirVector)
    R = []
    R.append(eta*a[0]*(eta * cosi - k**0.5) * normal[0])
    R.append(eta*a[1]*(eta * cosi - k**0.5) * normal[1])
    R.append(eta*a[2]*(eta * cosi - k**0.5) * normal[2])

    # R = eta * np.array(dirVector) + (eta * cosi - k**0.5) * normal
    return normalize(R)


def fresnel(normal, dirVector, ior):
    cosi = max(-1, min(1, dotProduct(dirVector, normal)))
    etai = 1
    etat = ior

    if cosi > 0:
        etai, etat = etat, etai

    sint = etai / etat * (max(0, 1 - cosi * cosi) ** 0.5)

    if sint >= 1:  # Total internal reflection
        return 1

    cost = max(0, 1 - sint * sint) ** 0.5
    cosi = abs(cosi)
    Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
    Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))

    return (Rs * Rs + Rp * Rp) / 2


class Raytracer(object):
    def __init__(self, width, height):
        # Constructor
        self.curr_color = (1, 1, 1)
        self.clear_color = (0, 0, 0)
        self.glViewMatrix()
        self.active_shader = None
        self.glCreateWindow(width, height)
        self.active_texture = None
        self.camPosition = V3(0, 0, 0)
        self.fov = 60

        self.background = None

        self.scene = []

        self.pointLights = []
        self.ambLight = None
        self.dirLight = None

        self.envmap = None

    def glFinish(self, filename):
        # Crea un archivo BMP y lo llena con la información dentro de self.pixels
        with open(filename, "wb") as file:
            # Header
            file.write(bytes('B'.encode('ascii')))
            file.write(bytes('M'.encode('ascii')))
            file.write(dword(14 + 40 + (self.width * self.height * 3)))
            file.write(dword(0))
            file.write(dword(14 + 40))

            # InfoHeader
            file.write(dword(40))
            file.write(dword(self.width))
            file.write(dword(self.height))
            file.write(word(1))
            file.write(word(24))
            file.write(dword(0))
            file.write(dword(self.width * self.height * 3))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))

            # Color Table
            for y in range(self.height):
                for x in range(self.width):
                    file.write(_color(self.pixels[x][y][0],
                                      self.pixels[x][y][1],
                                      self.pixels[x][y][2]))

    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.glClear()
        self.glViewport(0, 0, width, height)

    def glViewport(self, x, y, width, height):
        self.vpX = int(x)
        self.vpY = int(y)
        self.vpWidth = int(width)
        self.vpHeight = int(height)
        self.viewPortMatrix = [[width/2, 0, 0, x + width / 2],
                               [0, height/2, 0,  y + height / 2],
                               [0, 0, 0.5, 0.5],
                               [0, 0, 0, 1]]

        self.glProjectionMatrix()

    def glClearColor(self, r, g, b):
        self.clear_color = (r, g, b)

    def glClear(self):
        # Crea una lista 2D de pixeles y a cada valor le asigna 3 bytes de color
        self.pixels = [[self.clear_color for y in range(self.height)]
                       for x in range(self.width)]
        self.zbuffer = [[float('inf')for y in range(self.height)]
                        for x in range(self.width)]

    def glClearBackground(self):
        if self.background:
            for x in range(self.vpX, self.vpX + self.vpWidth):
                for y in range(self.vpY, self.vpY + self.vpHeight):

                    tx = (x - self.vpX) / self.vpWidth
                    ty = (y - self.vpY) / self.vpHeight

                    self.glPoint(x, y, self.background.getColor(tx, ty))

    def glViewportClear(self, color=None):
        for x in range(self.vpX, self.vpX + self.vpWidth):
            for y in range(self.vpY, self.vpY + self.vpHeight):
                self.glPoint(x, y, color)

    def glColor(self, r, g, b):
        self.curr_color = (r, g, b)

    def glPoint(self, x, y, color=None):
        if x < self.vpX or x >= self.vpX + self.vpWidth or y < self.vpY or y >= self.vpY + self.vpHeight:
            return

        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[int(x)][int(y)] = color or self.curr_color

    def glRender(self):

        for y in range(0, self.height, STEPS):
            for x in range(0, self.width, STEPS):
                # pasar de coordenadas de ventana a coordenadas NDC (-1 a 1)
                Px = 2 * ((x + 0.5) / self.width) - 1
                Py = 2 * ((y + 0.5) / self.height) - 1

                # Angulo de vision, asumiendo que el near plane esta a 1 unidad de la camara
                t = tan((self.fov * pi() / 180) / 2)
                r = t * self.width / self.height

                Px *= r
                Py *= t

                # La camara siempre esta viendo hacia -Z
                direction = V3(Px, Py, -1)
                # direction / np.linalg.norm(direction)
                direction = normalize(direction)

                self.glPoint(x, y, self.cast_ray(self.camPosition, direction))

    def scene_intersect(self, orig, dir, origObj=None):
        depth = float('inf')
        intersect = None

        for obj in self.scene:
            if obj is not origObj:
                hit = obj.ray_intersect(orig, dir)
                if hit != None:
                    if hit.distance < depth:
                        depth = hit.distance
                        intersect = hit

        return intersect

    def cast_ray(self, orig, dir, origObj=None, recursion=0):
        intersect = self.scene_intersect(orig, dir, origObj)

        if intersect == None or recursion >= MAX_RECURSION_DEPTH:
            if self.envmap:
                return self.envmap.getColor(dir)
            return self.clear_color

        material = intersect.sceneObject.material

        # Colors
        finalColor = [0, 0, 0]
        objectColor = [material.diffuse[0],
                       material.diffuse[1],
                       material.diffuse[2]]

        ambientColor = [0, 0, 0]
        dirLightColor = [0, 0, 0]
        pLightColor = [0, 0, 0]
        finalSpecColor = [0, 0, 0]
        reflectColor = [0, 0, 0]
        refractColor = [0, 0, 0]

        # Direccion de vista
        view_dir = subtract(self.camPosition, intersect.point)
        view_dir = normalize(view_dir)  # view_dir / np.linalg.norm(view_dir)

        if self.ambLight:
            ambientColor = list(self.ambLight.getColor())

        if self.dirLight:
            diffuseColor = [0, 0, 0]
            specColor = [0, 0, 0]
            shadow_intensity = 0

            # Iluminacion difusa
            # np.array(self.dirLight.direction) * -1
            light_dir = matriz_negada(list(self.dirLight.direction))
            intensity = max(0, dotProduct(intersect.normal, light_dir)
                            ) * self.dirLight.intensity
            diffuseColor = [intensity * self.dirLight.color[0],
                            intensity * self.dirLight.color[1],
                            intensity * self.dirLight.color[2]]

            # Iluminacion especular
            reflect = reflectVector(intersect.normal, light_dir)
            spec_intensity = self.dirLight.intensity * \
                max(0, dotProduct(view_dir, reflect)) ** material.spec
            specColor = [spec_intensity * self.dirLight.color[0],
                         spec_intensity * self.dirLight.color[1],
                         spec_intensity * self.dirLight.color[2]]

            # Shadow
            shadInter = self.scene_intersect(
                intersect.point, light_dir, intersect.sceneObject)
            if shadInter:
                shadow_intensity = 1

            dirLightColor = (1 - shadow_intensity) * diffuseColor
            finalSpecColor = add(
                finalSpecColor, (1 - shadow_intensity) * specColor)

        for pointLight in self.pointLights:
            diffuseColor = [0, 0, 0]
            specColor = [0, 0, 0]
            shadow_intensity = 0

            # Iluminacion difusa
            light_dir = subtract(pointLight.position, intersect.point)
            # light_dir / np.linalg.norm(light_dir)
            light_dir = normalize(light_dir)
            intensity = max(0, dotProduct(intersect.normal, light_dir)
                            ) * pointLight.intensity
            diffuseColor = [intensity * pointLight.color[0],
                            intensity * pointLight.color[1],
                            intensity * pointLight.color[2]]

            # Iluminacion especular
            reflect = reflectVector(intersect.normal, light_dir)
            spec_intensity = pointLight.intensity * \
                max(0, dotProduct(view_dir, reflect)) ** material.spec
            specColor = [spec_intensity * pointLight.color[0],
                         spec_intensity * pointLight.color[1],
                         spec_intensity * pointLight.color[2]]

            # Shadows
            shadInter = self.scene_intersect(
                intersect.point, light_dir, intersect.sceneObject)
            # lightDistance = np.linalg.norm(np.subtract(
            #     pointLight.position, intersect.point))
            lightDistance = getVectorMagnitude(
                subtract(pointLight.position, intersect.point))
            if shadInter and shadInter.distance < lightDistance:
                shadow_intensity = 1
            # print("P_LIGHT_COLOR")
            # print(pLightColor)
            # print("SHADOW OPERACION")
            # print((1 - shadow_intensity) * diffuseColor)
            if ((1 - shadow_intensity) * diffuseColor) == 0:
                pLightColor = add(
                    pLightColor, [0, 0, 0])
            else:
                pLightColor = add(
                    pLightColor, ((1 - shadow_intensity) * diffuseColor))
            # print("RESULTADO")
            # print(pLightColor)
            if((1 - shadow_intensity) * specColor) == 0:
                finalSpecColor = add(
                    finalSpecColor, [0, 0, 0])
            else:

                finalSpecColor = add(
                    finalSpecColor, (1 - shadow_intensity) * specColor)

        if material.matType == OPAQUE:

            res = []
          #  finalColor = pLightColor + ambientColor + dirLightColor + finalSpecColor
            if len(dirLightColor) != 3:
                dirLightColor = [0, 0, 0]

            res.append(pLightColor[0] + ambientColor[0] +
                       dirLightColor[0] + finalSpecColor[0])
            res.append(pLightColor[1] + ambientColor[1] +
                       dirLightColor[1] + finalSpecColor[1])
            res.append(pLightColor[2] + ambientColor[2] +
                       dirLightColor[2] + finalSpecColor[2])
            finalColor = res
            if material.texture and intersect.texCoords:
                texColor = material.texture.getColor(
                    intersect.texCoords[0], intersect.texCoords[1])

                res2 = []
                res2.append(texColor[0]*finalColor[0])
                res2.append(texColor[1]*finalColor[1])
                res2.append(texColor[2]*finalColor[2])

                finalColor = res2

        elif material.matType == REFLECTIVE:

            reflect = reflectVector(intersect.normal, matriz_negada(list(dir)))
            reflectColor = self.cast_ray(
                intersect.point, reflect, intersect.sceneObject, recursion + 1)
            reflectColor = list(reflectColor)
            res3 = []
            res3.append(reflectColor[0]+finalSpecColor[0])
            res3.append(reflectColor[1]+finalSpecColor[1])
            res3.append(reflectColor[2]+finalSpecColor[2])
            #finalColor = reflectColor + finalSpecColor
            finalColor = res3
        elif material.matType == TRANSPARENT:
            outside = dotProduct(dir, intersect.normal) < 0
            rbias = []
            rbias.append(0.001*intersect.normal[0])
            rbias.append(0.001*intersect.normal[1])
            rbias.append(0.001*intersect.normal[2])
            #bias = 0.001 * intersect.normal
            bias = rbias
            kr = fresnel(intersect.normal, dir, material.ior)

            # np.array(dir) * -1)
            reflect = reflectVector(intersect.normal, matriz_negada(list(dir)))
            reflectOrig = add(intersect.point, bias) if outside else subtract(
                intersect.point, bias)
            reflectColor = self.cast_ray(
                reflectOrig, reflect, None, recursion + 1)
            reflectColor = list(reflectColor)

            if kr < 1:
                refract = refractVector(intersect.normal, dir, material.ior)
                refractOrig = subtract(
                    intersect.point, bias) if outside else add(intersect.point, bias)
                refractColor = self.cast_ray(
                    refractOrig, refract, None, recursion + 1)
                refractColor = list(refractColor)
            res5 = []
            res5.append(reflectColor[0] * kr + refractColor[0]
                        * (1 - kr) + finalSpecColor[0])
            res5.append(reflectColor[1] * kr + refractColor[1]
                        * (1 - kr) + finalSpecColor[1])
            res5.append(reflectColor[2] * kr + refractColor[2]
                        * (1 - kr) + finalSpecColor[2])
           # finalColor = reflectColor * kr + refractColor * (1 - kr) + finalSpecColor
            finalColor = res5

        # Le aplicamos el color del objeto
        res6 = []
        res6.append(finalColor[0]*objectColor[0])
        res6.append(finalColor[1]*objectColor[1])
        res6.append(finalColor[2]*objectColor[2])
        finalColor = res6
       # finalColor *= objectColor

        # Nos aseguramos que no suba el valor de color de 1
        r = min(1, finalColor[0])
        g = min(1, finalColor[1])
        b = min(1, finalColor[2])

        return (r, g, b)

    def glTriangle_bc(self, A, B, C, texCoords=(), normals=(), verts=(),  color=None):
        # Bounding Box
        minX = round(min(A.x, B.x, C.x))
        minY = round(min(A.y, B.y, C.y))
        maxX = round(max(A.x, B.x, C.x))
        maxY = round(max(A.y, B.y, C.y))

        triangleNormal = crossProduct(subtract(
            verts[1], verts[0]), subtract(verts[2], verts[0]))
        triangleNormal = normalize(triangleNormal)

        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                u, v, w = baryCoords(A, B, C, V2(x, y))

                if u >= 0 and v >= 0 and w >= 0:

                    z = A.z * u + B.z * v + C.z * w

                    if 0 <= x < self.width and 0 <= y < self.height:
                        if z < self.zbuffer[x][y] and z <= 1 and z >= -1:

                            self.zbuffer[x][y] = z

                            if self.active_shader:

                                r, g, b = self.active_shader(self,
                                                             verts=verts,
                                                             baryCoords=(
                                                                 u, v, w),
                                                             texCoords=texCoords,
                                                             normals=normals,
                                                             triangleNormal=triangleNormal,
                                                             color=color or self.curr_color)

                                self.glPoint(x, y, _color(r, g, b))

                            else:
                                self.glPoint(x, y, color or self.curr_color)

    def glTransform(self, vertex, vMatrix):
        augVertex = V4(vertex[0], vertex[1], vertex[2], 1)

        transVertex = multiplyMatricesVector(vMatrix, augVertex)

        transVertex = V3(transVertex[0] / transVertex[3],
                         transVertex[1] / transVertex[3],
                         transVertex[2] / transVertex[3])

        return transVertex

    def glDirTransform(self, dirVector, vMatrix):
        augVertex = V4(dirVector[0], dirVector[1], dirVector[2], 0)
        transVertex = multiplyMatricesVector(vMatrix, augVertex)

        transVertex = V3(transVertex[0],
                         transVertex[1],
                         transVertex[2])

        return transVertex

    def glCamTransform(self, vertex):
        augVertex = V4(vertex[0], vertex[1], vertex[2], 1)

        transVertex = multiplyMatricesVector(multiplyMatrices(self.viewPortMatrix, multiplyMatrices(
            self.projectionMatrix, self.viewMatrix)), augVertex)

        transVertex = V3(transVertex[0] / transVertex[3],
                         transVertex[1] / transVertex[3],
                         transVertex[2] / transVertex[3])

        return transVertex

    def glCreateRotationMatrix(self, rotate=V3(0, 0, 0)):
        pitch = deg2rad(rotate.x)
        yaw = deg2rad(rotate.y)
        roll = deg2rad(rotate.z)
        rotationX = [[1, 0, 0, 0],
                     [0, cos(pitch), -sin(pitch), 0],
                     [0, sin(pitch), cos(pitch), 0],
                     [0, 0, 0, 1]]

        rotationY = [[cos(yaw), 0, sin(yaw), 0],
                     [0, 1, 0, 0],
                     [-sin(yaw), 0, cos(yaw), 0],
                     [0, 0, 0, 1]]

        rotationZ = [[cos(roll), -sin(roll), 0, 0],
                     [sin(roll), cos(roll), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]

        ab = multiplyMatrices(rotationX, rotationY)
        result = multiplyMatrices(ab, rotationZ)

        return result

    def glCreateObjectMatrix(self, translate=V3(0, 0, 0), scale=V3(1, 1, 1), rotate=V3(0, 0, 0)):

        translateMatrix = [[1, 0, 0, translate.x],
                           [0, 1, 0, translate.y],
                           [0, 0, 1, translate.z],
                           [0, 0, 0, 1]]

        scaleMatrix = [[scale.x, 0, 0, 0],
                       [0, scale.y, 0, 0],
                       [0, 0, scale.z, 0],
                       [0, 0, 0, 1]]

        rotationMatrix = self.glCreateRotationMatrix(rotate)

        ab = multiplyMatrices(translateMatrix, rotationMatrix)
        result = multiplyMatrices(ab, scaleMatrix)

        return result

    def glViewMatrix(self, translate=V3(0, 0, 0), rotate=V3(0, 0, 0)):
        self.camMatrix = self.glCreateObjectMatrix(
            translate, V3(1, 1, 1), rotate)
        self.viewMatrix = inverse(self.camMatrix)

    def glLookAt(self, eye, camPosition=V3(0, 0, 0)):
        forward = subtract(camPosition, eye)
        forward = normalize(forward)
        right = crossProduct(V3(0, 1, 0), forward)
        right = normalize(right)
        up = crossProduct(forward, right)
        up = normalize(up)

        camMatrix = [[right[0], up[0], forward[0], camPosition.x],
                     [right[1], up[1], forward[1], camPosition.y],
                     [right[2], up[2], forward[2], camPosition.z],
                     [0, 0, 0, 1]]

        self.viewMatrix = inverse(camMatrix)

    def glProjectionMatrix(self, n=0.1, f=1000, fov=60):
        t = tan((fov * pi() / 180) / 2) * n
        r = t * self.vpWidth / self.vpHeight

        self.projectionMatrix = [[n/r, 0, 0, 0],
                                 [0, n/t, 0, 0],
                                 [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                                 [0, 0, -1, 0]]

    def glLoadModel(self, filename, translate=V3(0, 0, 0), scale=V3(1, 1, 1), rotate=V3(0, 0, 0)):

        model = Obj(filename)

        modelMatrix = self.glCreateObjectMatrix(translate, scale, rotate)
        rotationMatrix = self.glCreateRotationMatrix(rotate)

        for face in model.faces:
            vertCount = len(face)

            vert0 = self.glTransform(
                model.vertices[face[0][0] - 1], modelMatrix)
            vert1 = self.glTransform(
                model.vertices[face[1][0] - 1], modelMatrix)
            vert2 = self.glTransform(
                model.vertices[face[2][0] - 1], modelMatrix)
            a = self.glCamTransform(vert0)
            b = self.glCamTransform(vert1)
            c = self.glCamTransform(vert2)

            vt0 = model.texcoords[face[0][1] - 1]
            vt1 = model.texcoords[face[1][1] - 1]
            vt2 = model.texcoords[face[2][1] - 1]

            vn0 = self.glDirTransform(
                model.normals[face[0][2] - 1], rotationMatrix)
            vn1 = self.glDirTransform(
                model.normals[face[1][2] - 1], rotationMatrix)
            vn2 = self.glDirTransform(
                model.normals[face[2][2] - 1], rotationMatrix)

            if vertCount == 4:
                vert3 = self.glTransform(
                    model.vertices[face[3][0] - 1], modelMatrix)
                d = self.glCamTransform(vert3)
                vt3 = model.texcoords[face[3][1] - 1]
                vn3 = self.glDirTransform(
                    model.normals[face[3][2] - 1], rotationMatrix)

            self.glTriangle_bc(a, b, c, texCoords=(vt0, vt1, vt2), normals=(
                vn0, vn1, vn2), verts=(vert0, vert1, vert2))

            if vertCount == 4:
                self.glTriangle_bc(a, c, d, texCoords=(vt0, vt2, vt3), normals=(
                    vn0, vn2, vn3), verts=(vert0, vert2, vert3))
