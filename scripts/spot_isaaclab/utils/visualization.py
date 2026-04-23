"""
Utilitários de visualização
"""


def apply_spot_colors(scene):
    """
    Aplica as cores originais do Spot (amarelo e preto) ao braço.
    
    Args:
        scene: InteractiveScene
    
    Returns:
        bool: True se bem-sucedido
    """
    print("[INFO] Aplicando cores originais do Spot no braço...")
    
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Sdf
        
        stage = omni.usd.get_context().get_stage()
        
        # Cores do Spot
        spot_yellow = Gf.Vec3f(0.988, 0.863, 0.247)  # #FCD93F
        spot_black = Gf.Vec3f(0.02, 0.02, 0.02)
        
        # Mapear links para cores
        arm_color_mapping = {
            "/World/envs/env_0/Robot/arm0_link_sh0": spot_black,
            "/World/envs/env_0/Robot/arm0_link_sh1": spot_yellow,
            "/World/envs/env_0/Robot/arm0_link_el0": spot_black,
            "/World/envs/env_0/Robot/arm0_link_el1": spot_yellow,
            "/World/envs/env_0/Robot/arm0_link_wr0": spot_yellow,
            "/World/envs/env_0/Robot/arm0_link_wr1": spot_black,
            "/World/envs/env_0/Robot/arm0_link_fngr": spot_black,
        }
        
        meshes_colored = 0
        
        for prim_path, color in arm_color_mapping.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            
            # Procurar e colorir meshes recursivamente
            def find_and_color_meshes(p, target_color):
                nonlocal meshes_colored
                
                if p.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(p)
                    # Método 1: DisplayColor
                    mesh.CreateDisplayColorAttr().Set([target_color])
                    
                    # Método 2: Material USD
                    try:
                        material_path = f"{p.GetPath()}/SpotMaterial"
                        material = UsdShade.Material.Define(stage, material_path)
                        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
                        shader.CreateIdAttr("UsdPreviewSurface")
                        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(target_color)
                        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
                        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
                        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                        UsdShade.MaterialBindingAPI(p).Bind(material)
                    except:
                        pass
                    
                    meshes_colored += 1
                
                for child in p.GetAllChildren():
                    find_and_color_meshes(child, target_color)
            
            find_and_color_meshes(prim, color)
        
        if meshes_colored > 0:
            print(f"[INFO] ✓ {meshes_colored} meshes coloridos com tema Spot!")
            return True
        else:
            print(f"[INFO] Usando cores originais do USD")
            return False
        
    except Exception as e:
        print(f"[INFO] Usando cores originais do USD: {e}")
        return False


class PerformanceMonitor:
    """Monitor de performance em tempo real"""
    
    def __init__(self):
        """Inicializa o monitor"""
        import time
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
    
    def update(self, step_count, extra_info=None):
        """
        Atualiza e exibe métricas.
        
        Args:
            step_count: Contador de steps
            extra_info: Dict com informações extras
        """
        import time
        
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            current_time = time.time()
            self.fps = 100 / (current_time - self.last_time)
            self.last_time = current_time
            
            # Montar string de info
            info_str = f"[PERF] FPS: {self.fps:.1f} | Steps: {step_count}"
            
            if extra_info:
                for key, value in extra_info.items():
                    if isinstance(value, float):
                        info_str += f" | {key}: {value:.2f}"
                    else:
                        info_str += f" | {key}: {value}"
            
            print(f"\r{info_str}", end="")
    
    def get_fps(self):
        """Retorna FPS atual"""
        return self.fps


def create_debug_sphere(stage, position, color=(1.0, 0.0, 0.0), radius=0.05, name="debug_sphere"):
    """
    Cria uma esfera de debug na cena.
    
    Args:
        stage: USD Stage
        position: tuple (x, y, z)
        color: tuple (r, g, b)
        radius: float
        name: string
    
    Returns:
        prim path
    """
    from pxr import UsdGeom, Gf
    
    sphere_path = f"/World/Debug/{name}"
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.CreateRadiusAttr().Set(radius)
    sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    
    xform = UsdGeom.Xformable(sphere)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    return sphere_path


def draw_trajectory_visualization(stage, waypoints, color=(0.0, 1.0, 0.0)):
    """
    Desenha trajetória como linha de pontos.
    
    Args:
        stage: USD Stage
        waypoints: Lista de pontos [(x,y,z), ...]
        color: tuple (r, g, b)
    """
    from pxr import UsdGeom, Gf, Vt
    
    # Criar curva
    curve_path = "/World/Debug/trajectory"
    curve = UsdGeom.BasisCurves.Define(stage, curve_path)
    
    # Pontos
    points = [Gf.Vec3f(*wp) for wp in waypoints]
    curve.CreatePointsAttr().Set(points)
    
    # Tipo
    curve.CreateTypeAttr().Set("linear")
    curve.CreateWrapAttr().Set("nonperiodic")
    
    # Cor
    curve.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    
    # Largura
    widths = [0.01] * len(waypoints)
    curve.CreateWidthsAttr().Set(widths)
    
    return curve_path