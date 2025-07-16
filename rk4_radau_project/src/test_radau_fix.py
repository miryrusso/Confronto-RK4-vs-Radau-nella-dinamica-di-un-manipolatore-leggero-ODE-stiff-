"""
test_radau_fix.py - Test per verificare la correzione dell'errore Radau

Questo script testa specificamente la chiamata a radau_scipy per assicurarsi
che funzioni correttamente con i parametri del manipolatore.
"""

import numpy as np
import sys

def test_radau_fix():
    """Test specifico per la correzione dell'errore Radau."""
    
    print("🔧 Test correzione errore Radau...")
    
    try:
        # Import dei moduli
        from dynamics import ManipulatorDynamics
        from integrators import NumericalIntegrators
        
        # Inizializzazione
        robot = ManipulatorDynamics()
        integrator = NumericalIntegrators(robot.manipulator_ode)
        
        # Parametri di test
        t_span = [0.0, 1.0]
        y0 = np.array([0.1, 0.1, 0.0, 0.0])
        k = 100.0
        
        print(f"  Parametri: t_span={t_span}, k={k}")
        print(f"  y0 = {y0}")
        
        # Test 1: Chiamata base di Radau
        print("\n  Test 1: Chiamata base Radau...")
        try:
            # CORRECTED: Tutti argomenti posizionali
            t1, y1, stats1 = integrator.radau_scipy(t_span, y0, 1e-6, 1e-9, k)
            
            if stats1.success:
                print(f"    ✓ Successo: {len(t1)} punti, {stats1.n_function_evaluations} nfev")
                print(f"    ✓ Stato finale: {y1[-1]}")
            else:
                print(f"    ✗ Fallimento: {stats1.error_message}")
                return False
                
        except Exception as e:
            print(f"    💥 Errore: {e}")
            return False
        
        # Test 2: Chiamata con tolleranze strette (come nel reference)
        print("\n  Test 2: Tolleranze strette per riferimento...")
        try:
            # CORRECTED: Tutti argomenti posizionali
            t2, y2, stats2 = integrator.radau_scipy(t_span, y0, 1e-10, 1e-12, k)
            
            if stats2.success:
                print(f"    ✓ Successo: {len(t2)} punti, {stats2.n_function_evaluations} nfev")
                print(f"    ✓ Stato finale: {y2[-1]}")
            else:
                print(f"    ✗ Fallimento: {stats2.error_message}")
                return False
                
        except Exception as e:
            print(f"    💥 Errore: {e}")
            return False
        
        # Test 3: Confronto con altri metodi
        print("\n  Test 3: Confronto con RK4...")
        try:
            t3, y3, stats3 = integrator.rk4_explicit(t_span, y0, 0.01, k)
            
            if stats3.success:
                error_comparison = np.linalg.norm(y1[-1] - y3[-1])
                print(f"    ✓ RK4 successo: {len(t3)} punti")
                print(f"    ✓ Differenza Radau-RK4: {error_comparison:.2e}")
            else:
                print(f"    ✗ RK4 fallimento: {stats3.error_message}")
                
        except Exception as e:
            print(f"    ⚠️  Errore RK4: {e}")
        
        # Test 4: Test ExperimentManager
        print("\n  Test 4: Test ExperimentManager...")
        try:
            from experiments import ExperimentManager
            
            exp_manager = ExperimentManager()
            # Configurazione ridotta per test veloce
            exp_manager.t_span = [0.0, 0.5]
            exp_manager.k_values = [100]
            
            # Test compute_reference_solution
            exp_manager.compute_reference_solution(100)
            print(f"    ✓ Soluzione di riferimento calcolata")
            
            # Test di un singolo esperimento
            result = exp_manager.test_adaptive_method(
                "Radau", exp_manager.integrator.radau_scipy, 100, 1e-6
            )
            
            if result['success']:
                print(f"    ✓ Test adaptive method: err={result.get('error_q1_rms', 'N/A'):.2e}")
            else:
                print(f"    ✗ Test adaptive method fallito: {result['error_message']}")
                
        except Exception as e:
            print(f"    💥 Errore ExperimentManager: {e}")
            return False
        
        print("\n✅ Tutti i test Radau superati!")
        return True
        
    except ImportError as e:
        print(f"💥 Errore di import: {e}")
        return False
    except Exception as e:
        print(f"💥 Errore generale: {e}")
        return False

def test_detailed_signature():
    """Test dettagliato della signature di radau_scipy."""
    
    print("\n🔍 Test dettagliato signature radau_scipy...")
    
    try:
        from integrators import NumericalIntegrators
        from dynamics import ManipulatorDynamics
        import inspect
        
        robot = ManipulatorDynamics()
        integrator = NumericalIntegrators(robot.manipulator_ode)
        
        # Ispezione della signature
        sig = inspect.signature(integrator.radau_scipy)
        print(f"  Signature: {sig}")
        
        # Test di varie chiamate
        t_span = [0.0, 0.1]
        y0 = np.array([0.1, 0.1, 0.0, 0.0])
        k = 50.0
        
        test_calls = [
            ("Tutto posizionale", lambda: integrator.radau_scipy(t_span, y0, 1e-6, 1e-9, k)),
            ("Solo rtol e k", lambda: integrator.radau_scipy(t_span, y0, 1e-6, None, k)),
            ("Rtol default", lambda: integrator.radau_scipy(t_span, y0, k=k)),  # questo potrebbe non funzionare
        ]
        
        for name, call_func in test_calls:
            try:
                t, y, stats = call_func()
                status = "✓" if stats.success else "✗"
                print(f"    {status} {name}: {len(t)} punti")
            except Exception as e:
                print(f"    💥 {name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"💥 Errore nel test signature: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TEST CORREZIONE ERRORE RADAU")
    print("=" * 60)
    
    success1 = test_radau_fix()
    success2 = test_detailed_signature()
    
    if success1 and success2:
        print("\n🎉 CORREZIONE VERIFICATA CON SUCCESSO!")
        print("Il problema con radau_scipy è stato risolto.")
        sys.exit(0)
    else:
        print("\n❌ PROBLEMI ANCORA PRESENTI")
        print("Verifica i messaggi di errore sopra.")
        sys.exit(1)