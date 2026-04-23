"""
Utilitários para inicialização suave do robô
"""
import torch


def initialize_robot_sequence(robot, scene, sim, arm_joint_indices):
    """
    Sequência de inicialização controlada:
    1. Pose Conhecida (Pernas em posição segura) -> 2. Estabilização.
    """
    # =========================================================================
    # PREPARAÇÃO
    # =========================================================================
    total_joints = robot.data.joint_pos.shape[1]
    leg_indices = [i for i in range(total_joints) if i not in arm_joint_indices]

    # Target final (Pose de operação padrão definida no config)
    target_full_pos = robot.data.default_joint_pos.clone()

    # --- VALORES FORNECIDOS PELO USUÁRIO (POSE INICIAL SEGURA) ---
    # Valores para as 12 juntas das pernas
    user_legs_values = [
         0.0288, -0.0251,  0.0387, -0.0320,  # Hips X 
         0.9778,  0.8721,  1.1695,  1.0712,  # Hips Y / Knees 
        -1.6483, -1.6556, -1.5027, -1.6000   # Knees / Ankles
    ]

    initial_curvature = {
        "shoulder": torch.pi/180.0*110.0,  # Curvatura inicial do ombro
        "elbow": torch.pi/180.0*45.0,     # Curvatura inicial do cot
    }
    initial_curvature["wrist"] = 180-((180-initial_curvature["shoulder"])+initial_curvature["elbow"])  # Curvatura inicial do pulso

    # Valores para as 7 juntas do braço
    user_arm_values = [
        0.0, -3.141+initial_curvature["shoulder"], # Shoulder 0/1
        3.141-initial_curvature["elbow"], 0.0,   # Elbow 0/1
        0.0-initial_curvature["wrist"], 0.0,   # Wrist 0/1
        -1.0        # Gripper
    ]
    
    
    # Aplicamos os valores do usuário nas juntas correspondentes
    if len(leg_indices) == len(user_legs_values):
        for i, leg_idx in enumerate(leg_indices):
            target_full_pos[0, leg_idx] = user_legs_values[i]
    else:
        print(f"[WARNING] Número de juntas de perna ({len(leg_indices)}) difere dos valores fornecidos ({len(user_legs_values)}). Usando default.")

    if len(arm_joint_indices) == len(user_arm_values):
        for j, arm_idx in enumerate(arm_joint_indices):
            target_full_pos[0, arm_idx] = user_arm_values[j]
    else:
        print(f"[WARNING] Número de juntas de braço ({len(arm_joint_indices)}) difere dos valores fornecidos ({len(user_arm_values)}). Usando default.")
    
    #print(f"[INFO] Pose inicial definida com valores do usuário para pernas: {start_pose} para índices {robot.data.joint_names}")
    # =========================================================================
    print("[INFO] Assumindo pose inicial segura...")
    
    # Define a posição inicial imediata para evitar "pulos" violentos se ele spawnou longe
    robot.write_joint_state_to_sim(target_full_pos, torch.zeros_like(target_full_pos)) 

    # =========================================================================
    # Estabilização Inicial
    # =========================================================================
    print("[INFO] Estabilizando robô...")
    for i in range(50):
        robot.set_joint_position_target(target_full_pos)
        
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(dt=sim.get_physics_dt())
        
        if i % 20 == 0:
            current_pos = robot.data.joint_pos
            error = torch.abs(current_pos - target_full_pos).max().item()
            if error < 0.05:
                print(f"[INFO] ✓ Robô estabilizado (Erro: {error:.4f} rad)")
                break
    
    robot.data.default_joint_pos = target_full_pos.clone()

    print("[INFO] ✓ Spot pronto para operação!")

def initialize_robot_sequence_multi_steps(robot, scene, sim, arm_joint_indices):
    """
    Sequência de inicialização controlada:
    1. Pose Conhecida (Pernas em posição segura) -> 2. Arrumar Braço -> 3. Transição para Pose Final.
    """
    # =========================================================================
    # PREPARAÇÃO
    # =========================================================================
    total_joints = robot.data.joint_pos.shape[1]
    leg_indices = [i for i in range(total_joints) if i not in arm_joint_indices]

    # Target final (Pose de operação padrão definida no config)
    target_full_pos = robot.data.default_joint_pos.clone()

    # --- VALORES FORNECIDOS PELO USUÁRIO (POSE INICIAL SEGURA) ---
    # Valores para as 12 juntas das pernas
    user_legs_values = [
         0.0288, -0.0251,  0.0387, -0.0320,  # Hips X (provavelmente)
         0.9778,  0.8721,  1.1695,  1.0712,  # Hips Y / Knees 
        -1.6483, -1.6556, -1.5027, -1.6000   # Knees / Ankles
    ]
    
    # Criamos um tensor para a pose inicial completa (misturando braço default + pernas custom)
    start_pose = robot.data.default_joint_pos.clone()
    
    # Aplicamos os valores do usuário apenas nos índices das pernas
    # Assumindo que a ordem de 'leg_indices' bate com a ordem do tensor fornecido
    # (Geralmente FL, FR, RL, RR ou similar, consistente com a simulação)
    if len(leg_indices) == len(user_legs_values):
        for i, leg_idx in enumerate(leg_indices):
            start_pose[0, leg_idx] = user_legs_values[i]
    else:
        print(f"[WARNING] Número de juntas de perna ({len(leg_indices)}) difere dos valores fornecidos ({len(user_legs_values)}). Usando default.")

    # =========================================================================
    # FASE 1: Assumir Pose Inicial (Hold)
    # O robô vai para a posição "em pé" fornecida imediatamente
    # =========================================================================
    print("[INFO] Fase 1: Assumindo pose inicial segura...")
    
    # Define a posição inicial imediata para evitar "pulos" violentos se ele spawnou longe
    # (Opcional: se o robô acabou de spawnar, podemos resetar o estado interno direto)
    # robot.write_joint_state_to_sim(pos=start_pose, vel=torch.zeros_like(start_pose)) 
    
    for i in range(300):
        # Força o target para a pose "start_pose" (Pernas custom + Braço default)
        robot.set_joint_position_target(start_pose)
        
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt=sim.get_physics_dt())

    print("[INFO] Pose inicial estabelecida.")

    # =========================================================================
    # FASE 2: Ajuste do Braço (Mantendo pernas fixas)
    # Interpola o braço da posição atual (que pode ter variado na física) para o target
    # Mantém as pernas travadas nos valores do usuário
    # =========================================================================
    print("[INFO] Fase 2: Ajustando braço...")
    
    # Captura onde o robô está agora (deve estar muito perto de start_pose)
    current_joint_pos = robot.data.joint_pos.clone()
    
    num_interp_steps_arm = 150
    
    for i in range(num_interp_steps_arm):
        alpha = min(1.0, (i + 1) / num_interp_steps_arm)
        alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        
        cmd_pos = start_pose.clone() # Base é a pose inicial das pernas
        
        # Interpola o braço (Current -> Default Target)
        for idx in arm_joint_indices:
            start = current_joint_pos[0, idx].item()
            end = target_full_pos[0, idx].item()
            cmd_pos[0, idx] = start + alpha_smooth * (end - start)
        
        robot.set_joint_position_target(cmd_pos)
        
        scene.write_data_to_sim()
        sim.step(render=(i % 10 == 0))
        scene.update(dt=sim.get_physics_dt())

    # =========================================================================
    # FASE 3: Transição das Pernas (Pose Inicial -> Pose Padrão)
    # O robô ajusta a postura das pernas dos valores manuais para o default do config
    # =========================================================================
    print("[INFO] Fase 3: Ajuste fino de postura (Pernas)...")
    
    num_interp_steps_legs = 200
    
    # O braço já está no target_full_pos. As pernas estão em start_pose (user values).
    
    for i in range(num_interp_steps_legs):
        # alpha = min(1.0, (i + 1) / num_interp_steps_legs)
        # alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        
        # cmd_pos = target_full_pos.clone() # Base agora é o target final
        
        # # Interpola as pernas (User Values -> Default)
        # for idx in leg_indices:
        #     start = start_pose[0, idx].item() # Começa nos valores manuais
        #     end = target_full_pos[0, idx].item() # Vai para o default
        #     cmd_pos[0, idx] = start + alpha_smooth * (end - start)
            
        # robot.set_joint_position_target(cmd_pos)
        
        scene.write_data_to_sim()
        sim.step(render=(i % 10 == 0))
        scene.update(dt=sim.get_physics_dt())

    # =========================================================================
    # FASE 4: Estabilização Final
    # =========================================================================
    print("[INFO] Fase 4: Estabilização final...")
    for i in range(100):
        #robot.set_joint_position_target(target_full_pos)
        
        scene.write_data_to_sim()
        sim.step(render=(i % 10 == 0))
        scene.update(dt=sim.get_physics_dt())
        
        if i % 20 == 0:
            current_pos = robot.data.joint_pos
            error = torch.abs(current_pos - target_full_pos).max().item()
            if error < 0.05:
                print(f"[INFO] ✓ Robô estabilizado (Erro: {error:.4f} rad)")
                break
    
    robot.data.default_joint_pos = target_full_pos.clone()

    print("[INFO] ✓ Spot pronto para operação!")

def get_arm_joint_info(robot):
    """
    Extrai informações das juntas do braço dinamicamente.
    Busca qualquer junta que comece com o prefixo 'arm'.
    
    Args:
        robot: Objeto Articulation
    
    Returns:
        tuple: (arm_joint_indices, arm_joint_names, target_positions)
    """
    arm_joint_indices = []
    arm_joint_names = []
    target_positions = []
    
    # Itera sobre todas as juntas do robô para encontrar as que queremos
    # robot.data.joint_names contém a lista ordenada de nomes
    for i, name in enumerate(robot.data.joint_names):
        # Verifica se o nome da junta começa com "arm"
        if name.startswith("arm"):
            arm_joint_indices.append(i)
            arm_joint_names.append(name)
            # Pega a posição padrão (default) para essa junta
            target_positions.append(robot.data.default_joint_pos[0, i].item())
    
    print(f"[INFO] Juntas do braço detectadas: {arm_joint_names}")
    
    # Verificação de segurança para evitar o erro de tensor vazio (RuntimeError)
    if not arm_joint_indices:
        print(f"[ERROR] Juntas disponíveis no robô: {robot.data.joint_names}")
        raise ValueError("Nenhuma junta começando com 'arm' foi encontrada! Verifique o prefixo no URDF.")

    return arm_joint_indices, arm_joint_names, target_positions