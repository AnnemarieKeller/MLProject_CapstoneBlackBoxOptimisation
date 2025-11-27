config = funcConfig.FUNCTION_CONFIG[functionNo]
print("Kernel type selected:", config.get("kernel_type"))
best_inputwithWHite, best_outputwithWhite, historyWithWhite= (bbo_loopWith(X_train, y_train, config, "EI"))
print("best_input" , best_inputwithWHite)
print( "-".join(f"{x:.6f}" for x in best_inputwithWHite))

we choose this for function 5 it generated the hightest inputs rbf kernel 
Iter 30 | Selected EI | Next input: [0.43081922 0.74894922 0.87284722 0.05651622] | Predicted y: -0.575806
GP best input: [0.74691695 0.97689221 0.95630257 0.97819293] value: 2214.0207709203423
SVR best input: [0.274937 0.803879 0.952871 0.996928] value: 2181.0684110226225
Dynamic best input: [0.43081922 0.74894922 0.87284722 0.05651622] value: -0.5758064718702058 but this was a bit better 



function 1 onfig = funcConfig.FUNCTION_CONFIG[functionNo]
print("Kernel type selected:", config.get("kernel_type"))
best_inputwithWHite, best_outputwithWhite, historyWithWhite= (bbo_loopWith(X_train, y_train, config, "EI"))
print("best_input" , best_inputwithWHite)
print( "-".join(f"{x:.6f}" for x in best_inputwithWHite))
print("best_output" , best_outputwithWhite)

