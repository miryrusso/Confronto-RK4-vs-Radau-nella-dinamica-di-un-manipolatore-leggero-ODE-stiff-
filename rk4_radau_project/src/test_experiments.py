from experiments import ExperimentManager

exp = ExperimentManager()
exp.t_span = [0.0, 2.0]  # tempo ridotto
exp.k_values = [100]     # una sola rigidezza  
exp.h_values = [0.01, 0.005]  # due passi
exp.rtol_values = [1e-4] # una sola tolleranza

print('🔬 Test esperimenti ridotto...')
exp.run_experiments()
print('✅ Test completato!')