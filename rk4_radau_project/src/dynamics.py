"""
dynamics.py - Modello dinamico del manipolatore planare 2-R con molle torsionali

Implementa le equazioni del moto derivate dal Lagrangiano per un braccio robotico
a due gradi di libertà con molle elastiche nei giunti.
"""

import numpy as np

class ManipulatorDynamics:
    """
    Classe per modellare la dinamica di un manipolatore planare 2-R con molle torsionali.
    
    Parametri fisici:
    - a1, a2: lunghezze dei link [m]
    - m1, m2: masse dei link [kg] 
    - I1, I2: momenti d'inerzia [kg⋅m²]
    - g: accelerazione gravitazionale [m/s²]
    - k: rigidezza delle molle torsionali [N⋅m/rad]
    """
    
    def __init__(self, a1=0.5, a2=0.4, m1=1.0, m2=0.8, g=9.81):
        self.a1 = a1    # lunghezza link 1 [m]
        self.a2 = a2    # lunghezza link 2 [m]
        self.m1 = m1    # massa link 1 [kg]
        self.m2 = m2    # massa link 2 [kg]
        self.g = g      # gravità [m/s²]
        
        # Momenti d'inerzia (barre omogenee sottili)
        self.I1 = m1 * a1**2 / 3
        self.I2 = m2 * a2**2 / 3
        
        # Posizioni dei centri di massa (a metà dei link)
        self.lc1 = a1 / 2
        self.lc2 = a2 / 2
    
    def mass_matrix(self, q):
        """
        Calcola la matrice di massa M(q) dipendente dalla configurazione.
        
        Args:
            q: array [θ1, θ2] angoli dei giunti [rad]
            
        Returns:
            M: matrice 2x2 di massa
        """
        theta1, theta2 = q[0], q[1]
        
        # Elementi della matrice di massa
        M11 = self.I1 + self.I2 + self.m2 * self.a1**2 + \
              2 * self.m2 * self.a1 * self.lc2 * np.cos(theta2)
              
        M12 = M21 = self.I2 + self.m2 * self.a1 * self.lc2 * np.cos(theta2)
        
        M22 = self.I2
        
        return np.array([[M11, M12],
                        [M21, M22]])
    
    def coriolis_matrix(self, q, q_dot):
        """
        Calcola la matrice di Coriolis C(q,q̇) e i termini centripeti.
        
        Args:
            q: array [θ1, θ2] angoli dei giunti [rad]
            q_dot: array [θ̇1, θ̇2] velocità angolari [rad/s]
            
        Returns:
            C_q_dot: vettore dei termini di Coriolis e centripeti
        """
        theta1, theta2 = q[0], q[1]
        theta1_dot, theta2_dot = q_dot[0], q_dot[1]
        
        # Coefficienti di Coriolis
        h = self.m2 * self.a1 * self.lc2 * np.sin(theta2)
        
        C_q_dot = np.array([
            -h * theta2_dot * (2 * theta1_dot + theta2_dot),
            h * theta1_dot**2
        ])
        
        return C_q_dot
    
    def gravity_vector(self, q):
        """
        Calcola il vettore gravitazionale G(q).
        
        Args:
            q: array [θ1, θ2] angoli dei giunti [rad]
            
        Returns:
            G: vettore 2x1 dei termini gravitazionali
        """
        theta1, theta2 = q[0], q[1]
        
        G1 = (self.m1 * self.lc1 + self.m2 * self.a1) * self.g * np.cos(theta1) + \
             self.m2 * self.lc2 * self.g * np.cos(theta1 + theta2)
             
        G2 = self.m2 * self.lc2 * self.g * np.cos(theta1 + theta2)
        
        return np.array([G1, G2])
    
    def spring_torques(self, q, k):
        """
        Calcola le coppie elastiche delle molle torsionali.
        
        Args:
            q: array [θ1, θ2] angoli dei giunti [rad]
            k: rigidezza delle molle [N⋅m/rad]
            
        Returns:
            K_q: vettore delle coppie elastiche
        """
        return k * q  # Molle lineari: τ = k⋅θ
    
    def manipulator_ode(self, t, y, k):
        """
        Funzione ODE principale per il sistema dinamico.
        
        Implementa: ẏ = f(t,y) dove y = [q, q̇]ᵀ
        
        Args:
            t: tempo corrente [s]
            y: stato [θ1, θ2, θ̇1, θ̇2]ᵀ
            k: rigidezza molle [N⋅m/rad]
            
        Returns:
            y_dot: derivata dello stato [θ̇1, θ̇2, θ̈1, θ̈2]ᵀ
        """
        # Estrazione delle variabili di stato
        q = y[:2]        # angoli [θ1, θ2]
        q_dot = y[2:]    # velocità [θ̇1, θ̇2]
        
        # Calcolo delle matrici dinamiche
        M = self.mass_matrix(q)
        C_q_dot = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)
        K_q = self.spring_torques(q, k)
        
        # Equazione del moto: M(q)q̈ + C(q,q̇)q̇ + G(q) + Kq = 0
        # Risolviamo per q̈: q̈ = -M⁻¹[C(q,q̇)q̇ + G(q) + Kq]
        try:
            q_ddot = -np.linalg.solve(M, C_q_dot + G + K_q)
        except np.linalg.LinAlgError:
            # Fallback in caso di matrice singolare
            q_ddot = -np.linalg.pinv(M) @ (C_q_dot + G + K_q)
        
        # Costruzione della derivata dello stato
        y_dot = np.concatenate([q_dot, q_ddot])
        
        return y_dot
    
    def kinematic_energy(self, y):
        """
        Calcola l'energia cinetica del sistema.
        
        Args:
            y: stato [θ1, θ2, θ̇1, θ̇2]ᵀ
            
        Returns:
            T: energia cinetica [J]
        """
        q = y[:2]
        q_dot = y[2:]
        
        M = self.mass_matrix(q)
        T = 0.5 * q_dot.T @ M @ q_dot
        
        return T
    
    def potential_energy(self, y, k):
        """
        Calcola l'energia potenziale totale (gravitazionale + elastica).
        
        Args:
            y: stato [θ1, θ2, θ̇1, θ̇2]ᵀ
            k: rigidezza molle [N⋅m/rad]
            
        Returns:
            V: energia potenziale [J]
        """
        q = y[:2]
        theta1, theta2 = q[0], q[1]
        
        # Energia potenziale gravitazionale
        h1 = self.lc1 * np.sin(theta1)  # altezza centro massa link 1
        h2 = self.a1 * np.sin(theta1) + self.lc2 * np.sin(theta1 + theta2)  # altezza centro massa link 2
        
        V_gravity = self.m1 * self.g * h1 + self.m2 * self.g * h2
        
        # Energia potenziale elastica (molle)
        V_spring = 0.5 * k * (theta1**2 + theta2**2)
        
        return V_gravity + V_spring
    
    def total_energy(self, y, k):
        """
        Calcola l'energia meccanica totale del sistema.
        
        Args:
            y: stato [θ1, θ2, θ̇1, θ̇2]ᵀ
            k: rigidezza molle [N⋅m/rad]
            
        Returns:
            E: energia totale [J]
        """
        return self.kinematic_energy(y) + self.potential_energy(y, k)
    
    def forward_kinematics(self, q):
        """
        Calcola la cinematica diretta dell'end-effector.
        
        Args:
            q: array [θ1, θ2] angoli dei giunti [rad]
            
        Returns:
            pos: posizione [x, y] dell'end-effector [m]
            orientation: orientamento θ1+θ2 [rad]
        """
        theta1, theta2 = q[0], q[1]
        
        x = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
        y = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)
        
        orientation = theta1 + theta2
        
        return np.array([x, y]), orientation

# Test della classe
if __name__ == "__main__":
    print("=== Test del modello dinamico del manipolatore ===")
    
    # Inizializzazione del sistema
    robot = ManipulatorDynamics()
    
    # Stato di test
    q_test = np.array([0.1, 0.1])  # [rad]
    q_dot_test = np.array([0.0, 0.0])  # [rad/s]
    y_test = np.concatenate([q_test, q_dot_test])
    k_test = 100.0  # [N⋅m/rad]
    
    print(f"Stato di test: q = {q_test}, q̇ = {q_dot_test}")
    print(f"Rigidezza molle: k = {k_test} N⋅m/rad")
    
    # Test delle matrici dinamiche
    M = robot.mass_matrix(q_test)
    C_q_dot = robot.coriolis_matrix(q_test, q_dot_test)
    G = robot.gravity_vector(q_test)
    K_q = robot.spring_torques(q_test, k_test)
    
    print(f"\nMatrice di massa M:\n{M}")
    print(f"Termini di Coriolis C(q,q̇)q̇: {C_q_dot}")
    print(f"Vettore gravitazionale G(q): {G}")
    print(f"Coppie elastiche K⋅q: {K_q}")
    
    # Test dell'ODE
    y_dot = robot.manipulator_ode(0.0, y_test, k_test)
    print(f"\nDerivata dello stato ẏ = f(t,y): {y_dot}")
    
    # Test delle energie
    T = robot.kinematic_energy(y_test)
    V = robot.potential_energy(y_test, k_test)
    E = robot.total_energy(y_test, k_test)
    
    print(f"\nEnergia cinetica T = {T:.6f} J")
    print(f"Energia potenziale V = {V:.6f} J")
    print(f"Energia totale E = {E:.6f} J")
    
    # Test cinematica diretta
    pos, orient = robot.forward_kinematics(q_test)
    print(f"\nPosizione end-effector: {pos} m")
    print(f"Orientamento end-effector: {orient:.4f} rad ({np.degrees(orient):.2f}°)")
    
    print("\n✓ Test completato con successo!")