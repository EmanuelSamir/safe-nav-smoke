import time
import numpy as np
import matplotlib.pyplot as plt
from hocbf_2nd_degree_analytical import HOCBF2ndDegreeAnalyticalController, simulate_dubins


if __name__ == "__main__":
    print("==============================================================")
    print("🎨 INICIANDO VISUALIZACIÓN DEL EFECTO DE PARÁMETROS HOCBF 🎨")
    print("==============================================================")
    
    # Parámetros comunes de simulación
    state_init = np.array([0.0, 0.0, 0.0]) # Inicio
    goal = np.array([10.0, 10.0])          # Meta
    
    # 3 Obstáculos desafiantes
    obstacles = [
        {'pos': [3.0, 2.2], 'r': 1.0},
        {'pos': [5.5, 5.5], 'r': 1.2},
        {'pos': [7.8, 8.0], 'r': 0.8}
    ]
    
    u_min = np.array([0.0, -4.0])
    u_max = np.array([3.0, 4.0])
    
    dt = 0.1
    steps = 180
    
    # Configurar los tres experimentos de ganancias
    parameter_runs = [
        {'name': 'Ganancias Altas (k=3.0)', 'k': 3.0, 'color': 'red', 'style': '--'},
        {'name': 'Ganancias Medias (k=1.2)', 'k': 1.2, 'color': 'orange', 'style': '-.'},
        {'name': 'Ganancias Suaves (k=0.5)', 'k': 0.5, 'color': 'green', 'style': '-'}
    ]
    
    results = {}
    
    for run in parameter_runs:
        print(f"\n🚀 Simulando: {run['name']} con k1=k2={run['k']}...")
        controller = HOCBF2ndDegreeAnalyticalController(u_min, u_max, d_safe=0.8, k1=run['k'], k2=run['k'])
        
        state = state_init.copy()
        trajectory = []
        reached = False
        times = []
        stops = 0
        
        for step in range(steps):
            trajectory.append(state.copy())
            
            t0 = time.perf_counter()
            u = controller.get_control(state, goal, obstacles)
            times.append(time.perf_counter() - t0)
            
            # Contar pasos en los que se detuvo por inviabilidad (velocidad = 0)
            if u[0] < 1e-3:
                stops += 1
                
            state = simulate_dubins(state, u, dt)
            
            if np.linalg.norm(state[:2] - goal) < 0.5:
                reached = True
                break
                
        trajectory = np.array(trajectory)
        results[run['name']] = {
            'traj': trajectory,
            'reached': reached,
            'time_ms': np.sum(times) * 1000.0,
            'avg_us': np.mean(times) * 1e6,
            'stops': stops,
            'length': len(trajectory)
        }
        
        print(f"   - Largo de trayectoria: {len(trajectory)} pasos")
        print(f"   - Pasos detenido por inviabilidad: {stops}")
        print(f"   - ¿Llegó a la meta?: {'SÍ' if reached else 'NO'}")

    # -------------------------------------------------------------
    # GENERAR GRÁFICO DE ALTA CALIDAD CON MATPLOTLIB
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # 1. Dibujar el inicio y la meta
    ax.scatter(state_init[0], state_init[1], color='blue', s=150, zorder=5, label='Inicio (0,0)')
    ax.scatter(goal[0], goal[1], color='gold', edgecolor='black', s=200, marker='*', zorder=5, label='Meta (10,10)')
    
    # 2. Dibujar los obstáculos como círculos rellenos
    for i, obs in enumerate(obstacles):
        circle_safe = plt.Circle(obs['pos'], obs['r'] + 0.8, color='grey', alpha=0.15, fill=True, label='Zona de Seguridad' if i==0 else "")
        circle_phys = plt.Circle(obs['pos'], obs['r'], color='black', alpha=0.6, fill=True, label='Obstáculo Físico' if i==0 else "")
        ax.add_patch(circle_safe)
        ax.add_patch(circle_phys)
        ax.text(obs['pos'][0], obs['pos'][1], f"Obs {i+1}", color='white', ha='center', va='center', fontweight='bold')

    # 3. Dibujar las trayectorias de los tres experimentos
    for run in parameter_runs:
        data = results[run['name']]
        traj = data['traj']
        ax.plot(
            traj[:, 0], traj[:, 1], 
            color=run['color'], linestyle=run['style'], linewidth=2.5, 
            label=f"{run['name']} - {'Llegó' if data['reached'] else 'No Llegó'} ({data['stops']} paradas)"
        )
        # Dibujar flechitas de orientación cada 20 pasos para visualizar la cinemática
        for idx in range(0, len(traj), 20):
            ax.quiver(
                traj[idx, 0], traj[idx, 1], 
                np.cos(traj[idx, 2]), np.sin(traj[idx, 2]), 
                color=run['color'], scale=15, width=0.005, alpha=0.6
            )

    # Configuración de los límites y detalles estéticos del gráfico
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.set_xlabel("Coordenada X (m)", fontsize=12)
    ax.set_ylabel("Coordenada Y (m)", fontsize=12)
    ax.set_title("Efecto de la Ganancia CBF (k1, k2) en la Evasión Slalom", fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_aspect('equal')
    
    # Guardar gráfico
    output_path = "/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/playground/cbf_parameter_comparison.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*65)
    print(f"📊 ¡ANÁLISIS DE RENDIMIENTO Y OPTIMALIDAD DE PARÁMETROS COMPLETADO! 📊")
    print("="*65)
    print(f"🎨 El gráfico comparativo ha sido guardado exitosamente en:")
    print(f"   [cbf_parameter_comparison.png]({output_path})")
    print("="*65)
