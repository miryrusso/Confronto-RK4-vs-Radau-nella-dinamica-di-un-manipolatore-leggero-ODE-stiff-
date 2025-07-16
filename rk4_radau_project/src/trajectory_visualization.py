"""
trajectory_visualization.py - Visualizzazione traiettorie e animazioni del manipolatore

Crea grafici delle traiettorie nel tempo, nello spazio delle configurazioni,
e animazioni del movimento del braccio robotico.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dynamics import ManipulatorDynamics
from integrators import NumericalIntegrators
import seaborn as sns

class TrajectoryVisualizer:
    """
    Classe per visualizzare le traiettorie del manipolatore e confrontare i metodi.
    """
    
    def __init__(self):
        self.robot = ManipulatorDynamics()
        self.integrator = NumericalIntegrators(self.robot.manipulator_ode)
        
        # Configurazione standard
        self.t_span = [0.0, 10.0]
        self.y0 = np.array([0.1, 0.1, 0.0, 0.0])  # [θ1, θ2, θ̇1, θ̇2]
    
    def simulate_trajectory(self, method_name, k, **kwargs):
        """
        Simula una traiettoria con un metodo specifico.
        
        Args:
            method_name: nome del metodo ('RK4', 'Radau', ecc.)
            k: rigidezza delle molle
            **kwargs: parametri specifici del metodo (h, rtol, ecc.)
        
        Returns:
            t, y, stats: tempo, stati, statistiche
        """
        if method_name == "RK4":
            h = kwargs.get('h', 0.01)
            return self.integrator.rk4_explicit(self.t_span, self.y0, h, k)
        
        elif method_name == "Radau":
            rtol = kwargs.get('rtol', 1e-6)
            return self.integrator.radau_scipy(self.t_span, self.y0, rtol, rtol*1e-3, k)
        
        elif method_name == "Euler_Explicit":
            h = kwargs.get('h', 0.001)
            return self.integrator.euler_explicit(self.t_span, self.y0, h, k)
        
        elif method_name == "Euler_Implicit":
            h = kwargs.get('h', 0.01)
            return self.integrator.euler_implicit(self.t_span, self.y0, h, k)
        
        elif method_name == "Trapezoid":
            h = kwargs.get('h', 0.01)
            return self.integrator.trapezoid_implicit(self.t_span, self.y0, h, k)
        
        else:
            raise ValueError(f"Metodo {method_name} non riconosciuto")
    
    def plot_joint_trajectories(self, k_values=[10, 100, 500], save_fig=True):
        """
        Grafica le traiettorie degli angoli dei giunti per diverse rigidezze.
        """
        fig, axes = plt.subplots(len(k_values), 2, figsize=(15, 4*len(k_values)))
        if len(k_values) == 1:
            axes = axes.reshape(1, -1)
        
        methods_config = [
            ("RK4", {'h': 0.005}, 'blue', '-'),
            ("Radau", {'rtol': 1e-6}, 'red', '--'),
        ]
        
        for i, k in enumerate(k_values):
            print(f"Simulando traiettorie per k = {k}...")
            
            for method_name, params, color, style in methods_config:
                try:
                    t, y, stats = self.simulate_trajectory(method_name, k, **params)
                    
                    if stats.success:
                        # Angolo θ1
                        axes[i, 0].plot(t, y[:, 0], color=color, linestyle=style, 
                                       label=f"{method_name}", linewidth=2)
                        
                        # Angolo θ2
                        axes[i, 1].plot(t, y[:, 1], color=color, linestyle=style, 
                                       label=f"{method_name}", linewidth=2)
                    else:
                        print(f"  {method_name} fallito: {stats.error_message}")
                        
                except Exception as e:
                    print(f"  Errore con {method_name}: {e}")
            
            # Configurazione degli assi
            axes[i, 0].set_title(f'θ₁(t) - k = {k} N⋅m/rad')
            axes[i, 0].set_xlabel('Tempo [s]')
            axes[i, 0].set_ylabel('θ₁ [rad]')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            
            axes[i, 1].set_title(f'θ₂(t) - k = {k} N⋅m/rad')
            axes[i, 1].set_xlabel('Tempo [s]')
            axes[i, 1].set_ylabel('θ₂ [rad]')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('joint_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_phase_portraits(self, k_values=[10, 100, 500], save_fig=True):
        """
        Crea i ritratti di fase (θ vs θ̇) per visualizzare la dinamica.
        """
        fig, axes = plt.subplots(len(k_values), 2, figsize=(15, 4*len(k_values)))
        if len(k_values) == 1:
            axes = axes.reshape(1, -1)
        
        for i, k in enumerate(k_values):
            print(f"Creando ritratti di fase per k = {k}...")
            
            # Simula con Radau (alta qualità)
            t, y, stats = self.simulate_trajectory("Radau", k, rtol=1e-8)
            
            if stats.success:
                # Ritratto di fase per θ1
                axes[i, 0].plot(y[:, 0], y[:, 2], 'b-', linewidth=2, alpha=0.8)
                axes[i, 0].plot(y[0, 0], y[0, 2], 'go', markersize=8, label='Inizio')
                axes[i, 0].plot(y[-1, 0], y[-1, 2], 'ro', markersize=8, label='Fine')
                axes[i, 0].set_xlabel('θ₁ [rad]')
                axes[i, 0].set_ylabel('θ̇₁ [rad/s]')
                axes[i, 0].set_title(f'Ritratto di fase θ₁ - k = {k} N⋅m/rad')
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].legend()
                
                # Ritratto di fase per θ2
                axes[i, 1].plot(y[:, 1], y[:, 3], 'b-', linewidth=2, alpha=0.8)
                axes[i, 1].plot(y[0, 1], y[0, 3], 'go', markersize=8, label='Inizio')
                axes[i, 1].plot(y[-1, 1], y[-1, 3], 'ro', markersize=8, label='Fine')
                axes[i, 1].set_xlabel('θ₂ [rad]')
                axes[i, 1].set_ylabel('θ̇₂ [rad/s]')
                axes[i, 1].set_title(f'Ritratto di fase θ₂ - k = {k} N⋅m/rad')
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].legend()
            else:
                print(f"  Simulazione fallita per k = {k}")
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('phase_portraits.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_end_effector_trajectory(self, k_values=[10, 100, 500], save_fig=True):
        """
        Grafica la traiettoria dell'end-effector nello spazio cartesiano.
        """
        fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 5))
        if len(k_values) == 1:
            axes = [axes]
        
        for i, k in enumerate(k_values):
            print(f"Calcolando traiettoria end-effector per k = {k}...")
            
            # Simula con alta precisione
            t, y, stats = self.simulate_trajectory("Radau", k, rtol=1e-8)
            
            if stats.success:
                # Calcola posizioni end-effector
                positions = np.array([self.robot.forward_kinematics(q)[0] for q in y[:, :2]])
                
                # Grafica la traiettoria
                axes[i].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.8)
                axes[i].plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Inizio')
                axes[i].plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='Fine')
                
                # Workspace del manipolatore
                circle_outer = plt.Circle((0, 0), self.robot.a1 + self.robot.a2, 
                                        fill=False, linestyle='--', alpha=0.5, color='gray')
                circle_inner = plt.Circle((0, 0), abs(self.robot.a1 - self.robot.a2), 
                                        fill=False, linestyle='--', alpha=0.5, color='gray')
                axes[i].add_patch(circle_outer)
                axes[i].add_patch(circle_inner)
                
                axes[i].set_xlabel('x [m]')
                axes[i].set_ylabel('y [m]')
                axes[i].set_title(f'Traiettoria end-effector\nk = {k} N⋅m/rad')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                axes[i].set_aspect('equal')
                
                # Imposta limiti appropriati
                margin = 0.1
                axes[i].set_xlim(-self.robot.a1 - self.robot.a2 - margin, 
                               self.robot.a1 + self.robot.a2 + margin)
                axes[i].set_ylim(-self.robot.a1 - self.robot.a2 - margin, 
                               self.robot.a1 + self.robot.a2 + margin)
            else:
                print(f"  Simulazione fallita per k = {k}")
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('end_effector_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_energy_evolution(self, k_values=[10, 100, 500], save_fig=True):
        """
        Grafica l'evoluzione dell'energia nel tempo per diversi metodi.
        """
        fig, axes = plt.subplots(len(k_values), 1, figsize=(12, 4*len(k_values)))
        if len(k_values) == 1:
            axes = [axes]
        
        methods_config = [
            ("RK4", {'h': 0.005}, 'blue', '-'),
            ("Radau", {'rtol': 1e-6}, 'red', '--'),
            ("Euler_Explicit", {'h': 0.001}, 'green', ':'),
        ]
        
        for i, k in enumerate(k_values):
            print(f"Analizzando evoluzione energetica per k = {k}...")
            
            for method_name, params, color, style in methods_config:
                try:
                    t, y, stats = self.simulate_trajectory(method_name, k, **params)
                    
                    if stats.success and len(t) > 10:
                        # Calcola energia totale
                        energies = np.array([self.robot.total_energy(state, k) for state in y])
                        
                        # Normalizza rispetto al valore iniziale
                        E0 = energies[0]
                        energy_relative = (energies - E0) / abs(E0) if abs(E0) > 1e-12 else energies - E0
                        
                        axes[i].plot(t, energy_relative, color=color, linestyle=style, 
                                   label=f"{method_name}", linewidth=2)
                    else:
                        print(f"  {method_name} fallito o troppo pochi punti")
                        
                except Exception as e:
                    print(f"  Errore con {method_name}: {e}")
            
            axes[i].set_xlabel('Tempo [s]')
            axes[i].set_ylabel('(E - E₀) / E₀')
            axes[i].set_title(f'Conservazione energia - k = {k} N⋅m/rad')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('energy_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def animate_manipulator(self, k=100, method="Radau", save_animation=False, **kwargs):
        """
        Crea un'animazione del movimento del manipolatore.
        """
        print(f"Creando animazione con {method}, k = {k}...")
        
        # Simula la traiettoria
        t, y, stats = self.simulate_trajectory(method, k, **kwargs)
        
        if not stats.success:
            print(f"Simulazione fallita: {stats.error_message}")
            return
        
        # Sottocampiona per l'animazione (max 200 frames)
        n_frames = min(200, len(t))
        indices = np.linspace(0, len(t)-1, n_frames, dtype=int)
        t_anim = t[indices]
        y_anim = y[indices]
        
        # Setup della figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Animazione del manipolatore
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Manipolatore 2-R (k = {k} N⋅m/rad)')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        
        # Elementi grafici del manipolatore
        link1_line, = ax1.plot([], [], 'b-', linewidth=6, label='Link 1')
        link2_line, = ax1.plot([], [], 'r-', linewidth=6, label='Link 2')
        joint1_point, = ax1.plot([], [], 'ko', markersize=10)
        joint2_point, = ax1.plot([], [], 'ko', markersize=8)
        end_effector_point, = ax1.plot([], [], 'ro', markersize=8, label='End-effector')
        trajectory_line, = ax1.plot([], [], 'g--', alpha=0.5, linewidth=1)
        
        ax1.legend()
        
        # Subplot 2: Evoluzione degli angoli
        ax2.plot(t, y[:, 0], 'b-', label='θ₁', linewidth=2)
        ax2.plot(t, y[:, 1], 'r-', label='θ₂', linewidth=2)
        current_time_line = ax2.axvline(x=t[0], color='black', linestyle='--', label='Tempo corrente')
        ax2.set_xlabel('Tempo [s]')
        ax2.set_ylabel('Angoli [rad]')
        ax2.set_title('Evoluzione angoli dei giunti')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Dati per la traiettoria dell'end-effector
        end_effector_positions = []
        
        def animate(frame):
            # Stato corrente
            q = y_anim[frame, :2]
            theta1, theta2 = q[0], q[1]
            
            # Posizioni dei giunti
            x1 = self.robot.a1 * np.cos(theta1)
            y1 = self.robot.a1 * np.sin(theta1)
            
            x2 = x1 + self.robot.a2 * np.cos(theta1 + theta2)
            y2 = y1 + self.robot.a2 * np.sin(theta1 + theta2)
            
            # Aggiorna grafica del manipolatore
            link1_line.set_data([0, x1], [0, y1])
            link2_line.set_data([x1, x2], [y1, y2])
            joint1_point.set_data([x1], [y1])
            joint2_point.set_data([x2], [y2])
            end_effector_point.set_data([x2], [y2])
            
            # Traiettoria dell'end-effector
            end_effector_positions.append([x2, y2])
            if len(end_effector_positions) > 1:
                trajectory = np.array(end_effector_positions)
                trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
            
            # Aggiorna linea del tempo corrente
            current_time_line.set_xdata([t_anim[frame], t_anim[frame]])
            
            return [link1_line, link2_line, joint1_point, joint2_point, 
                   end_effector_point, trajectory_line, current_time_line]
        
        # Crea l'animazione
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                     interval=50, blit=True, repeat=True)
        
        if save_animation:
            try:
                anim.save('manipulator_animation.gif', writer='pillow', fps=20)
                print("Animazione salvata come 'manipulator_animation.gif'")
            except Exception as e:
                print(f"Errore nel salvare l'animazione: {e}")
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def compare_methods_stability(self, k=500, h_values=None, save_fig=True):
        """
        Confronta la stabilità dei metodi espliciti per valori crescenti di h.
        """
        if h_values is None:
            h_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        methods = ["Euler_Explicit", "RK4"]
        colors = ['blue', 'red']
        
        fig, axes = plt.subplots(len(methods), 1, figsize=(12, 6*len(methods)))
        if len(methods) == 1:
            axes = [axes]
        
        print(f"Testando stabilità per k = {k} N⋅m/rad...")
        
        for i, method in enumerate(methods):
            print(f"  Metodo: {method}")
            
            for j, h in enumerate(h_values):
                try:
                    t, y, stats = self.simulate_trajectory(method, k, h=h)
                    
                    if stats.success:
                        # Verifica se la simulazione è rimasta stabile
                        max_angle = np.max(np.abs(y[:, :2]))
                        if max_angle < 10:  # angoli ragionevoli (< 10 rad)
                            axes[i].plot(t, y[:, 0], color=colors[i], alpha=0.7, 
                                       label=f'h = {h:.3f}' if j < 5 else "")
                        else:
                            axes[i].plot(t[:int(len(t)*0.8)], y[:int(len(t)*0.8), 0], 
                                       color='red', linestyle='--', alpha=0.5,
                                       label=f'h = {h:.3f} (instabile)' if j < 3 else "")
                    else:
                        print(f"    h = {h:.3f}: FALLITO ({stats.error_message})")
                        
                except Exception as e:
                    print(f"    h = {h:.3f}: ERRORE ({e})")
            
            axes[i].set_xlabel('Tempo [s]')
            axes[i].set_ylabel('θ₁ [rad]')
            axes[i].set_title(f'{method} - Stabilità vs passo h (k = {k} N⋅m/rad)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        if save_fig:
            plt.savefig('stability_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# Script di test per la visualizzazione
if __name__ == "__main__":
    print("=== Test del visualizzatore di traiettorie ===")
    
    # Inizializza il visualizzatore
    viz = TrajectoryVisualizer()
    
    print("1. Grafici delle traiettorie degli angoli...")
    viz.plot_joint_trajectories(k_values=[100])
    
    print("2. Ritratti di fase...")
    viz.plot_phase_portraits(k_values=[100])
    
    print("3. Traiettoria end-effector...")
    viz.plot_end_effector_trajectory(k_values=[100])
    
    print("4. Evoluzione energia...")
    viz.plot_energy_evolution(k_values=[100])
    
    print("5. Confronto stabilità...")
    viz.compare_methods_stability(k=500)
    
    # Test animazione (solo se richiesto)
    create_animation = input("\nCreare animazione? (y/n): ").lower() == 'y'
    if create_animation:
        print("6. Creazione animazione...")
        anim = viz.animate_manipulator(k=100, method="Radau", rtol=1e-6)
    
    print("\n✓ Test di visualizzazione completato!")