"""
test_modules.py - Test unitari per verificare il corretto funzionamento di ogni modulo

Esegue test sistematici per ogni componente del progetto prima di lanciare
gli esperimenti completi.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def test_dynamics_module():
    """Test del modulo dynamics.py"""
    print("🔧 Test del modulo dynamics...")
    
    try:
        from dynamics import ManipulatorDynamics
        
        # Inizializzazione
        robot = ManipulatorDynamics()
        print(f"  ✓ Manipolatore inizializzato: a1={robot.a1}m, a2={robot.a2}m")
        
        # Test stato di esempio
        q = np.array([0.1, 0.2])  # angoli [rad]
        q_dot = np.array([0.5, -0.3])  # velocità [rad/s]
        y = np.concatenate([q, q_dot])
        k = 100.0  # rigidezza [N⋅m/rad]
        
        # Test matrice di massa
        M = robot.mass_matrix(q)
        assert M.shape == (2, 2), "Matrice massa ha dimensione errata"
        assert np.allclose(M, M.T), "Matrice massa non simmetrica"
        assert np.all(np.linalg.eigvals(M) > 0), "Matrice massa non definita positiva"
        print(f"  ✓ Matrice di massa: det(M) = {np.linalg.det(M):.3f}")
        
        # Test termini di Coriolis
        C_q_dot = robot.coriolis_matrix(q, q_dot)
        assert C_q_dot.shape == (2,), "Vettore Coriolis ha dimensione errata"
        print(f"  ✓ Termini Coriolis: ||C|| = {np.linalg.norm(C_q_dot):.3f}")
        
        # Test gravità
        G = robot.gravity_vector(q)
        assert G.shape == (2,), "Vettore gravità ha dimensione errata"
        print(f"  ✓ Vettore gravità: ||G|| = {np.linalg.norm(G):.3f}")
        
        # Test molle
        K_q = robot.spring_torques(q, k)
        expected_K_q = k * q
        assert np.allclose(K_q, expected_K_q), "Termini elastici errati"
        print(f"  ✓ Coppie elastiche: K⋅q = {K_q}")
        
        # Test ODE
        y_dot = robot.manipulator_ode(0.0, y, k)
        assert y_dot.shape == (4,), "ODE restituisce vettore di dimensione errata"
        assert np.allclose(y_dot[:2], q_dot), "Prime due componenti ODE errate"
        print(f"  ✓ Sistema ODE: ẏ = {y_dot}")
        
        # Test energie
        T = robot.kinematic_energy(y)
        V = robot.potential_energy(y, k)
        E = robot.total_energy(y, k)
        
        assert T >= 0, "Energia cinetica negativa"
        assert np.isclose(E, T + V), "Energia totale inconsistente"
        print(f"  ✓ Energie: T={T:.3f}J, V={V:.3f}J, E={E:.3f}J")
        
        # Test cinematica diretta
        pos, orient = robot.forward_kinematics(q)
        assert pos.shape == (2,), "Posizione end-effector ha dimensione errata"
        assert isinstance(orient, (int, float, np.number)), "Orientamento non scalare"
        
        # Verifica che la posizione sia nel workspace
        dist_from_origin = np.linalg.norm(pos)
        min_reach = abs(robot.a1 - robot.a2)
        max_reach = robot.a1 + robot.a2
        assert min_reach <= dist_from_origin <= max_reach, "Posizione fuori workspace"
        print(f"  ✓ Cinematica: pos={pos}, orient={orient:.3f}rad")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Errore nel test dynamics: {e}")
        return False

def test_integrators_module():
    """Test del modulo integrators.py"""
    print("🔧 Test del modulo integrators...")
    
    try:
        from integrators import NumericalIntegrators
        
        # Sistema di test: oscillatore armonico
        def harmonic_oscillator(t, y, omega=1.0):
            """y'' + ω²y = 0 -> [y, y'] -> [y', -ω²y]"""
            return np.array([y[1], -omega**2 * y[0]])
        
        integrator = NumericalIntegrators(harmonic_oscillator)
        
        # Parametri di test
        t_span = [0.0, 2*np.pi]  # un periodo
        y0 = np.array([1.0, 0.0])  # condizioni iniziali
        omega = 1.0
        h = 0.1  # passo di integrazione
        rtol = 1e-6  # tolleranza
        
        # Soluzione analitica
        def exact_solution(t):
            return np.array([np.cos(omega * t), -omega * np.sin(omega * t)])
        
        methods_to_test = [
            ("Euler Esplicito", lambda: integrator.euler_explicit(t_span, y0, h, omega)),
            ("Euler Implicito", lambda: integrator.euler_implicit(t_span, y0, h, omega)),
            ("Trapezio", lambda: integrator.trapezoid_implicit(t_span, y0, h, omega)),
            ("RK4", lambda: integrator.rk4_explicit(t_span, y0, h, omega)),
            ("Radau", lambda: integrator.radau_scipy(t_span, y0, rtol, rtol*1e-3, omega))
        ]
        
        print("  Test su oscillatore armonico...")
        results = {}
        
        for name, method_func in methods_to_test:
            try:
                start_time = time.time()
                t, y, stats = method_func()
                elapsed = time.time() - start_time
                
                if stats.success and len(t) > 1:
                    # Calcola errore finale
                    y_exact_final = exact_solution(t[-1])
                    error = np.linalg.norm(y[-1] - y_exact_final)
                    
                    results[name] = {
                        'success': True,
                        'error': error,
                        'nfev': stats.n_function_evaluations,
                        'time': elapsed,
                        'n_points': len(t)
                    }
                    
                    print(f"    ✓ {name:15}: err={error:.2e}, nfev={stats.n_function_evaluations:3d}, t={elapsed:.3f}s")
                else:
                    results[name] = {'success': False, 'error_msg': stats.error_message}
                    print(f"    ✗ {name:15}: FALLITO ({stats.error_message})")
                    
            except Exception as e:
                results[name] = {'success': False, 'error_msg': str(e)}
                print(f"    ✗ {name:15}: ERRORE ({e})")
        
        # Verifica che almeno alcuni metodi funzionino
        successful_methods = [name for name, res in results.items() if res.get('success', False)]
        if len(successful_methods) >= 2:
            print(f"  ✓ {len(successful_methods)}/5 metodi funzionanti")
            return True
        else:
            print(f"  ✗ Solo {len(successful_methods)}/5 metodi funzionanti")
            return False
        
    except Exception as e:
        print(f"  ✗ Errore nel test integrators: {e}")
        return False

def test_experiments_module():
    """Test del modulo experiments.py (versione ridotta)"""
    print("🔧 Test del modulo experiments...")
    
    try:
        from experiments import ExperimentManager
        
        # Crea un manager con configurazione ridotta
        exp_manager = ExperimentManager()
        
        # Modifica configurazione per test rapido
        exp_manager.t_span = [0.0, 1.0]  # tempo ridotto
        exp_manager.k_values = [100]  # una sola rigidezza
        exp_manager.h_values = [0.01, 0.005]  # pochi passi
        exp_manager.rtol_values = [1e-4]  # una sola tolleranza
        
        print("  Test configurazione esperimenti...")
        print(f"    Intervallo: {exp_manager.t_span}")
        print(f"    Rigidezze: {exp_manager.k_values}")
        print(f"    Passi h: {exp_manager.h_values}")
        print(f"    Tolleranze: {exp_manager.rtol_values}")
        
        # Test calcolo soluzione di riferimento
        k_test = 100
        try:
            exp_manager.compute_reference_solution(k_test)
            print(f"  ✓ Soluzione di riferimento calcolata per k={k_test}")
        except Exception as e:
            print(f"  ✗ Errore nella soluzione di riferimento: {e}")
            return False
        
        # Test di un singolo metodo
        try:
            result = exp_manager.test_fixed_step_method(
                "RK4", exp_manager.integrator.rk4_explicit, k_test, 0.01
            )
            
            if result['success']:
                print(f"  ✓ Test singolo metodo RK4: err={result.get('error_q1_rms', 'N/A'):.2e}")
            else:
                print(f"  ✗ Test singolo metodo fallito: {result['error_message']}")
                return False
                
        except Exception as e:
            print(f"  ✗ Errore nel test singolo metodo: {e}")
            return False
        
        # Test calcolo metriche di energia
        try:
            t_ref, y_ref = exp_manager.reference_solutions[k_test]
            energy_metrics = exp_manager.compute_energy_drift(t_ref, y_ref, k_test)
            print(f"  ✓ Metriche energia: drift={energy_metrics['energy_drift_relative']:.2e}")
        except Exception as e:
            print(f"  ✗ Errore nelle metriche energia: {e}")
            return False
        
        print("  ✓ Modulo experiments funziona correttamente")
        return True
        
    except Exception as e:
        print(f"  ✗ Errore nel test experiments: {e}")
        return False

def test_visualization_module():
    """Test del modulo trajectory_visualization.py"""
    print("🔧 Test del modulo visualization...")
    
    try:
        from trajectory_visualization import TrajectoryVisualizer
        
        viz = TrajectoryVisualizer()
        print("  ✓ Visualizzatore inizializzato")
        
        # Test simulazione traiettoria
        try:
            t, y, stats = viz.simulate_trajectory("RK4", k=100, h=0.01)
            
            if stats.success:
                print(f"  ✓ Simulazione traiettoria: {len(t)} punti, {stats.n_function_evaluations} eval")
            else:
                print(f"  ✗ Simulazione fallita: {stats.error_message}")
                return False
                
        except Exception as e:
            print(f"  ✗ Errore nella simulazione: {e}")
            return False
        
        # Test calcolo posizioni end-effector
        try:
            positions = np.array([viz.robot.forward_kinematics(q)[0] for q in y[:5, :2]])
            assert positions.shape == (5, 2), "Posizioni end-effector hanno forma errata"
            print(f"  ✓ Calcolo posizioni end-effector: prima pos = {positions[0]}")
        except Exception as e:
            print(f"  ✗ Errore nel calcolo posizioni: {e}")
            return False
        
        # Test creazione grafico semplice (senza mostrarlo)
        try:
            plt.ioff()  # Disabilita display interattivo
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(t[:100], y[:100, 0], 'b-', label='θ₁')
            ax.set_xlabel('Tempo [s]')
            ax.set_ylabel('θ₁ [rad]')
            ax.legend()
            ax.grid(True)
            plt.close(fig)  # Chiudi senza mostrare
            print("  ✓ Test creazione grafico completato")
        except Exception as e:
            print(f"  ✗ Errore nella creazione grafico: {e}")
            return False
        
        print("  ✓ Modulo visualization funziona correttamente")
        return True
        
    except Exception as e:
        print(f"  ✗ Errore nel test visualization: {e}")
        return False

def test_integration_workflow():
    """Test dell'integrazione tra tutti i moduli"""
    print("🔧 Test workflow integrato...")
    
    try:
        from dynamics import ManipulatorDynamics
        from integrators import NumericalIntegrators
        
        # Workflow completo: dynamics -> integrators -> analisi
        robot = ManipulatorDynamics()
        integrator = NumericalIntegrators(robot.manipulator_ode)
        
        # Parametri di test
        t_span = [0.0, 2.0]
        y0 = np.array([0.1, 0.1, 0.0, 0.0])
        k_values = [50, 200]
        
        results = {}
        
        for k in k_values:
            print(f"    Test integrato con k = {k}...")
            
            # Test RK4
            t_rk4, y_rk4, stats_rk4 = integrator.rk4_explicit(t_span, y0, 0.01, k)
            
            # Test Radau
            t_radau, y_radau, stats_radau = integrator.radau_scipy(t_span, y0, 1e-6, 1e-9, k)
            
            if stats_rk4.success and stats_radau.success:
                # Confronto energie
                E_rk4_initial = robot.total_energy(y_rk4[0], k)
                E_rk4_final = robot.total_energy(y_rk4[-1], k)
                E_radau_initial = robot.total_energy(y_radau[0], k)
                E_radau_final = robot.total_energy(y_radau[-1], k)
                
                drift_rk4 = abs(E_rk4_final - E_rk4_initial) / abs(E_rk4_initial)
                drift_radau = abs(E_radau_final - E_radau_initial) / abs(E_radau_initial)
                
                results[k] = {
                    'rk4_points': len(t_rk4),
                    'radau_points': len(t_radau),
                    'rk4_drift': drift_rk4,
                    'radau_drift': drift_radau,
                    'rk4_nfev': stats_rk4.n_function_evaluations,
                    'radau_nfev': stats_radau.n_function_evaluations
                }
                
                print(f"      RK4: {len(t_rk4)} punti, drift={drift_rk4:.2e}, nfev={stats_rk4.n_function_evaluations}")
                print(f"      Radau: {len(t_radau)} punti, drift={drift_radau:.2e}, nfev={stats_radau.n_function_evaluations}")
            else:
                print(f"      ✗ Fallimento per k={k}")
                return False
        
        # Verifica coerenza dei risultati
        print("  ✓ Workflow integrato completato con successo")
        
        # Verifica tendenze attese
        k_low, k_high = k_values[0], k_values[1]
        if results[k_high]['rk4_nfev'] > results[k_low]['rk4_nfev']:
            print("  ✓ Trend atteso: RK4 richiede più eval per k maggiori")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Errore nel test workflow integrato: {e}")
        return False

def run_performance_benchmark():
    """Esegue un benchmark delle performance"""
    print("⚡ Benchmark delle performance...")
    
    try:
        from dynamics import ManipulatorDynamics
        from integrators import NumericalIntegrators
        
        robot = ManipulatorDynamics()
        integrator = NumericalIntegrators(robot.manipulator_ode)
        
        # Parametri benchmark
        t_span = [0.0, 1.0]
        y0 = np.array([0.1, 0.1, 0.0, 0.0])
        k = 100
        n_runs = 5
        
        methods = [
            ("RK4", lambda: integrator.rk4_explicit(t_span, y0, 0.01, k)),
            ("Radau", lambda: integrator.radau_scipy(t_span, y0, 1e-6, 1e-9, k))
        ]
        
        print(f"  Benchmark su {n_runs} esecuzioni...")
        
        for name, method_func in methods:
            times = []
            nfevs = []
            
            for i in range(n_runs):
                start_time = time.time()
                t, y, stats = method_func()
                elapsed = time.time() - start_time
                
                if stats.success:
                    times.append(elapsed)
                    nfevs.append(stats.n_function_evaluations)
                else:
                    print(f"    ✗ {name} fallito alla run {i+1}")
                    return False
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_nfev = np.mean(nfevs)
            
            print(f"    {name:10}: {avg_time:.4f}±{std_time:.4f}s, {avg_nfev:.0f} eval")
        
        print("  ✓ Benchmark completato")
        return True
        
    except Exception as e:
        print(f"  ✗ Errore nel benchmark: {e}")
        return False

def main():
    """Esegue tutti i test dei moduli"""
    print("=" * 60)
    print("TEST UNITARI DEI MODULI DEL PROGETTO")
    print("=" * 60)
    print()
    
    tests = [
        ("Dynamics", test_dynamics_module),
        ("Integrators", test_integrators_module),
        ("Experiments", test_experiments_module),
        ("Visualization", test_visualization_module),
        ("Integration", test_integration_workflow),
        ("Performance", run_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        start_time = time.time()
        
        try:
            success = test_func()
            elapsed = time.time() - start_time
            results[test_name] = {'success': success, 'time': elapsed}
            
            if success:
                print(f"✅ {test_name} completato in {elapsed:.2f}s")
            else:
                print(f"❌ {test_name} fallito")
                
        except Exception as e:
            elapsed = time.time() - start_time
            results[test_name] = {'success': False, 'time': elapsed, 'error': str(e)}
            print(f"💥 {test_name} errore: {e}")
    
    # Riepilogo finale
    print("\n" + "="*60)
    print("RIEPILOGO TEST")
    print("="*60)
    
    total_tests = len(tests)
    passed_tests = sum(1 for result in results.values() if result['success'])
    total_time = sum(result['time'] for result in results.values())
    
    print(f"\n📊 Risultati: {passed_tests}/{total_tests} test superati")
    print(f"⏱️  Tempo totale: {total_time:.2f}s")
    
    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {test_name:<15}: {result['time']:.2f}s")
        if not result['success'] and 'error' in result:
            print(f"       Errore: {result['error']}")
    
    if passed_tests == total_tests:
        print("\n🎉 TUTTI I TEST SUPERATI! Il progetto è pronto per l'esecuzione completa.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test falliti. Risolvi i problemi prima di procedere.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)