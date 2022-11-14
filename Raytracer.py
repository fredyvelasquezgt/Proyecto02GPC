from gl import Raytracer, V3
from obj import *
from libraryGame import Renderer
from figures import *


width = 1024
height = 512

mirror = Material(spec=128, matType=REFLECTIVE)

glass = Material(spec=64, ior=1.5, matType=TRANSPARENT)
diamond = Material(spec=64, ior=2.417, matType=TRANSPARENT)

adorno = Material(texture=Texture('textures/bombillaRoja.bmp'))
suelo = Material(texture=Texture('textures/snow.bmp'))
snow = Material(texture=Texture('textures/snow.bmp'))

piedra = Material(texture=Texture('textures/piedra.bmp'))
regalo1 = Material(texture=Texture('textures/gift.bmp'))
regalo2 = Material(texture=Texture('textures/regalo2.bmp'))
regalo3 = Material(texture=Texture('textures/regalo3.bmp'))
regalo4 = Material(texture=Texture('textures/regalo4.bmp'))
regalo5 = Material(texture=Texture('textures/regalo5.bmp'))
regalo6 = Material(texture=Texture('textures/regalo6.bmp'))


rtx = Raytracer(width, height)
rtx.envmap = EnvMap('backgrounds/fondo.bmp')

rtx.ambLight = AmbientLight(strength=0.1)
rtx.dirLight = DirectionalLight(direction=V3(1, -1, -2), intensity=0.5)
rtx.pointLights.append(PointLight(position=V3(0, 2, 0), intensity=0.5))

# Objetos
# Snowman
rtx.scene.append(Sphere(V3(0, -3, -8), 1, snow))
rtx.scene.append(Sphere(V3(0, -2, -8), 0.75, snow))
rtx.scene.append(Sphere(V3(0, -1, -8), 0.5, snow))
rtx.scene.append(Sphere(V3(0, -3, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(0, -2, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(0, -2.5, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(0, -1, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(0, -1.5, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(0, -3, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(0.2, -0.9, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(-0.2, -0.9, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(0.2, -0.6, -7), 0.1, piedra))
rtx.scene.append(Sphere(V3(-0.2, -0.6, -7), 0.1, piedra))
# Regalos
rtx.scene.append(AABB(V3(1, -3.5, -7), V3(0.5, 0.5, 0.5), regalo4))
rtx.scene.append(AABB(V3(2, -3.5, -7), V3(0.5, 0.5, 0.5), regalo1))
rtx.scene.append(AABB(V3(3, -3.5, -7), V3(0.5, 0.5, 0.5), regalo2))
rtx.scene.append(AABB(V3(-5, -3.5, -8), V3(0.5, 0.5, 0.5), regalo3))
rtx.scene.append(AABB(V3(3, -3.5, -7), V3(0.5, 0.5, 0.5), regalo5))
rtx.scene.append(AABB(V3(3, -3.5, -8), V3(0.5, 0.5, 0.5), regalo6))
# Floor
rtx.scene.append(AABB(V3(0, -4, -8), V3(20, 0.1, 5), suelo))
# Bombillas
rtx.scene.append(Sphere(V3(10, 3, -8), 0.5, glass))
rtx.scene.append(Sphere(V3(8, 3, -8), 0.5, glass))
rtx.scene.append(Sphere(V3(6, 3, -8), 0.5, glass))
rtx.scene.append(Sphere(V3(4, 3, -8), 0.5, glass))
rtx.scene.append(Sphere(V3(2, 3, -8), 0.5, glass))
rtx.scene.append(Sphere(V3(0, 3, -9), 0.5, glass))
rtx.scene.append(Sphere(V3(-2, 3, -8), 0.5, glass))
rtx.scene.append(Sphere(V3(-4, 3, -7), 0.5, glass))
rtx.scene.append(Sphere(V3(-6, 3, -6), 0.5, glass))

rtx.glRender()
# rtx.active_texture = Texture('textures/tree.bmp')
# rtx.glLoadModel("obj/tree3.obj",
#                 translate=V3(0, 0, -8),
#                 scale=V3(0.75, 0.75, 0.75),
#                 rotate=V3(0, 67, 0))

# Terminar

rtx.glFinish('Proyecto2.bmp')
