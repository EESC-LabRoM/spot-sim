"""
Utilitários para configuração de colisões
"""


def create_primitive_arm_colliders(scene, sim):
    """
    Cria colisores primitivos (cápsulas) para o braço do Spot.
    Mais eficiente que usar meshes detalhados.
    
    Args:
        scene: InteractiveScene
        sim: SimulationContext
    
    Returns:
        bool: True se bem-sucedido
    """
    print("[INFO] Criando colisores primitivos para o braço...")
    
    try:
        import omni.usd
        from pxr import UsdPhysics, UsdGeom
        
        stage = omni.usd.get_context().get_stage()
        
        # Especificações dos colisores (path, raio, altura)
        collider_specs = [
            ("/World/envs/env_0/Robot/arm0_link_sh0", 0.08, 0.15),
            ("/World/envs/env_0/Robot/arm0_link_sh1", 0.06, 0.25),
            ("/World/envs/env_0/Robot/arm0_link_el0", 0.05, 0.25),
            ("/World/envs/env_0/Robot/arm0_link_el1", 0.04, 0.15),
            ("/World/envs/env_0/Robot/arm0_link_wr0", 0.04, 0.12),
            ("/World/envs/env_0/Robot/arm0_link_wr1", 0.04, 0.10),
            ("/World/envs/env_0/Robot/arm0_link_fngr", 0.035, 0.15),
        ]
        
        colliders_created = 0
        
        for link_path, radius, height in collider_specs:
            link_prim = stage.GetPrimAtPath(link_path)
            if not link_prim.IsValid():
                continue
            
            # Criar cápsula
            capsule_path = f"{link_path}/collision_capsule"
            capsule = UsdGeom.Capsule.Define(stage, capsule_path)
            capsule.CreateRadiusAttr().Set(radius)
            capsule.CreateHeightAttr().Set(height)
            capsule.CreateAxisAttr().Set("Z")
            
            # Adicionar física
            capsule_prim = stage.GetPrimAtPath(capsule_path)
            UsdPhysics.CollisionAPI.Apply(capsule_prim)
            
            # Tornar invisível
            imageable = UsdGeom.Imageable(capsule_prim)
            imageable.CreateVisibilityAttr().Set("invisible")
            
            colliders_created += 1
        
        if colliders_created > 0:
            print(f"[INFO] ✓ {colliders_created} colisores criados!")
            # Forçar update
            for _ in range(10):
                sim.step(render=False)
            return True
        
        return False
        
    except Exception as e:
        print(f"[ERRO] Falha ao criar colisores: {e}")
        import traceback
        traceback.print_exc()
        return False


def enable_mesh_collisions(scene, sim):
    """
    Habilita colisões detalhadas baseadas em meshes.
    Mais preciso mas mais custoso computacionalmente.
    
    Args:
        scene: InteractiveScene
        sim: SimulationContext
    
    Returns:
        bool: True se bem-sucedido
    """
    print("[INFO] Habilitando colisões de mesh do braço...")
    
    try:
        import omni.usd
        from pxr import UsdPhysics, UsdGeom
        
        stage = omni.usd.get_context().get_stage()
        
        arm_links = [
            "/World/envs/env_0/Robot/arm0_link_sh0",
            "/World/envs/env_0/Robot/arm0_link_sh1",
            "/World/envs/env_0/Robot/arm0_link_el0", 
            "/World/envs/env_0/Robot/arm0_link_el1",
            "/World/envs/env_0/Robot/arm0_link_wr0",
            "/World/envs/env_0/Robot/arm0_link_wr1",
            "/World/envs/env_0/Robot/arm0_link_fngr",
        ]
        
        collision_enabled = 0
        
        for link_path in arm_links:
            prim = stage.GetPrimAtPath(link_path)
            if not prim.IsValid():
                continue
            
            # Adicionar CollisionAPI ao link
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            
            # Processar meshes recursivamente
            def process_mesh_collisions(p):
                nonlocal collision_enabled
                
                if p.IsA(UsdGeom.Mesh):
                    if not p.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI.Apply(p)
                    
                    if not p.HasAPI(UsdPhysics.MeshCollisionAPI):
                        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(p)
                        mesh_collision.CreateApproximationAttr().Set("convexHull")
                        collision_enabled += 1
                
                for child in p.GetAllChildren():
                    process_mesh_collisions(child)
            
            process_mesh_collisions(prim)
        
        if collision_enabled > 0:
            print(f"[INFO] ✓ {collision_enabled} colisores de mesh habilitados!")
            for _ in range(5):
                sim.step(render=False)
            return True
        
        return False
        
    except Exception as e:
        print(f"[ERRO] Falha ao habilitar colisões: {e}")
        import traceback
        traceback.print_exc()
        return False