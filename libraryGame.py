import struct
from collections import namedtuple


from obj import Obj


from numpy import sin, cos, tan


from myLibrary import *

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


def _color(r, g, b):
    # Acepta valores de 0 a 1
    # Se asegura que la información de color se guarda solamente en 3 bytes
    return bytes([int(b * 255), int(g * 255), int(r * 255)])


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


BLACK = _color(0, 0, 0)
WHITE = _color(1, 1, 1)


class Renderer(object):
    def __init__(self, width, height):
        # Constructor
        self.curr_color = WHITE
        self.clear_color = BLACK
        self.glViewMatrix()
        self.glCreateWindow(width, height)
        self.active_texture = None
        self.active_texture2 = None
        self.normal_map = None
        self.background = None
        self.active_shader = None
        self.directional_light = V3(0, 0, -1)

    def glClearBackground(self):
        if self.background:
            for x in range(self.vpX, self.vpX + self.vpWidth):
                for y in range(self.vpY, self.vpY + self.vpHeight):

                    tx = (x - self.vpX) / self.vpWidth
                    ty = (y - self.vpY) / self.vpHeight

                    self.glPoint(x, y, self.background.getColor(tx, ty))

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
                    file.write(self.pixels[x][y])

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
        self.clear_color = _color(r, g, b)

    def glClear(self):
        # Crea una lista 2D de pixeles y a cada valor le asigna 3 bytes de color
        self.pixels = [[self.clear_color for y in range(self.height)]
                       for x in range(self.width)]

        self.zbuffer = [[float('inf')for y in range(self.height)]
                        for x in range(self.width)]

    def glViewportClear(self, color=None):
        for x in range(self.vpX, self.vpX + self.vpWidth):
            for y in range(self.vpY, self.vpY + self.vpHeight):
                self.glPoint(x, y, color)

    def glColor(self, r, g, b):
        self.curr_color = _color(r, g, b)

    def glPoint(self, x, y, color=None):
        if x < self.vpX or x >= self.vpX + self.vpWidth or y < self.vpY or y >= self.vpY + self.vpHeight:
            return

        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[int(x)][int(y)] = color or self.curr_color

    def glPoint_NDC(self, x, y, color=None):
        x = int((x + 1) * (self.vpWidth / 2) + self.vpX)
        y = int((y + 1) * (self.vpHeight / 2) + self.vpY)

        if x < self.vpX or x >= self.vpX + self.vpWidth or y < self.vpY or y >= self.vpY + self.vpHeight:
            return

        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[int(x)][int(y)] = color or self.curr_color

    def glLine(self, v0, v1, color=None):
        x0 = v0.x
        x1 = v1.x
        y0 = v0.y
        y1 = v1.y

        if x0 == x1 and y0 == y1:
            self.glPoint(x0, y1, color)
            return

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        offset = 0
        limit = 0.5
        m = dy/dx
        y = y0

        for x in range(x0, x1 + 1):
            if steep:
                self.glPoint(y, x, color)
            else:
                self.glPoint(x, y, color)

            offset += m
            if offset >= limit:
                y += 1 if y0 < y1 else -1
                limit += 1

    def glLine_NDC(self, v0, v1, color=None):

        x0 = int((v0.x + 1) * (self.vpWidth / 2) + self.vpX)
        x1 = int((v1.x + 1) * (self.vpWidth / 2) + self.vpX)
        y0 = int((v0.y + 1) * (self.vpHeight / 2) + self.vpY)
        y1 = int((v1.y + 1) * (self.vpHeight / 2) + self.vpY)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        offset = 0
        limit = 0.5
        m = dy/dx
        y = y0

        for x in range(x0, x1 + 1):
            if steep:
                self.glPoint(y, x, color)
            else:
                self.glPoint(x, y, color)

            offset += m
            if offset >= limit:
                y += 1 if y0 < y1 else -1
                limit += 1

    def glTriangle_standard(self, A, B, C, color=None):

        if A.y < B.y:
            A, B = B, A
        if A.y < C.y:
            A, C = C, A
        if B.y < C.y:
            B, C = C, B

        def flatBottomTriangle(v1, v2, v3):
            try:
                d_21 = (v2.x - v1.x) / (v2.y - v1.y)
                d_31 = (v3.x - v1.x) / (v3.y - v1.y)
            except:
                pass
            else:
                x1 = v2.x
                x2 = v3.x
                for y in range(v2.y, v1.y + 1):
                    self.glLine(V2(int(x1), y), V2(int(x2), y), color)
                    x1 += d_21
                    x2 += d_31

        def flatTopTriangle(v1, v2, v3):
            try:
                d_31 = (v3.x - v1.x) / (v3.y - v1.y)
                d_32 = (v3.x - v2.x) / (v3.y - v2.y)
            except:
                pass
            else:
                x1 = v3.x
                x2 = v3.x

                for y in range(v3.y, v1.y + 1):
                    self.glLine(V2(int(x1), y), V2(int(x2), y), color)
                    x1 += d_31
                    x2 += d_32

        if B.y == C.y:
            # triangulo con base inferior plana
            flatBottomTriangle(A, B, C)
        elif A.y == B.y:
            # triangulo con base superior plana
            flatTopTriangle(A, B, C)
        else:
            # dividir el triangulo en dos
            # dibujar ambos casos
            # Teorema de intercepto
            D = V2(A.x + ((B.y - A.y) / (C.y - A.y)) * (C.x - A.x), B.y)
            flatBottomTriangle(A, B, D)
            flatTopTriangle(B, D, C)

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
       # print(result)
       # print(translateMatrix * rotationMatrix * scaleMatrix)
        return result
       # return(translateMatrix * rotationMatrix * scaleMatrix)

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
