from enum import Enum, auto

class TextureType(Enum):
    gray_texture = "./textures/gray_textures/"
    wood_texture = "./textures/wood_textures/"
    crawled_texture = "./textures/crawled_textures_crop/"
    mixed_texture = "./textures/mixed_crawled_textures/" 

class ArrangeType(Enum):
    align = 0                

class TableTextureType(Enum):
    black = TextureType.gray_texture
    mixed = TextureType.mixed_texture

class WoodTextureType(Enum):
    wood = TextureType.wood_texture
    nowood = TextureType.gray_texture

class CameraViewType(Enum): # relative to world frame
    graspspace = [(1.1, -0.35, 0.1), (1.3, -0.15, 0.16)]
    workspace = [(0.4, -0.1, 0.1), (0.6, 0.1, 0.16)]

class CameraLocation(Enum): # relative to world frame
    top = [(0, -0.3, 1.6), (0.7, 0.3, 1.8)]
    left = [(0.3, -0.8, 1), (1.6, -0.3, 1.4)]
    right = [(0.1, 0.3, 1), (0.8, 0.8, 1.4)]