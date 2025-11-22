week 7 Function 2 	
Function 3	best_output 1.0095803268953585
	0.021216-0.186651-0.953626best_output 1.0067536995455515
0.469938-0.362513-0.245139
Function 4 	0.950000-0.950000-0.950000-0.950000 - function 4 
Function7  0.662104-0.603671-0.744697-0.956912-0.781182-0.974278
Function 5 	
0.071846-0.990240-0.965780-0.979771

Function 6 	
best_input [0.15348443 0.87381027 0.2628882  0.75404657 0.02718982]
0.153484-0.873810-0.262888-0.754047-0.027190
best_output -0.6426717459237652 est_input [0.36701811 0.53047771 0.12654971 0.77753932 0.03362224]
best_output -0.6422946414765327
0.367018-0.530478-0.126550-0.777539-0.033622 Iter 30 | Selected UCB | Next input: [0.2432077  0.31125229 0.07104966 0.73999137 0.0163741 ] | Predicted y: -0.676756
GP best input: [0.52124909 0.96582931 0.03323703 0.70339929 0.01121944] value: -0.571538358300738
0.521249-0.965829-0.033237-0.703399-0.011219
SVR best input: [0.45052702 0.38006693 0.54301609 0.81017291 0.12901649] value: -0.5420068891776895
0.450527-0.380067-0.543016-0.810173-0.129016
Dymnaic best input: [0.28561404 0.54168267 0.11029461 0.74263229 0.02570542] value: 1.5334715101773249

Function 7 	

/0.662104-0.603671-0.744697-0.956912-0.781182-0.974278
Function 8 	built dynamic 
Iter 30: Optimized kernel = Matern(length_scale=[1.65, 2.54, 1.19, 2.72, 6.62, 11, 1.71, 11], nu=2.5) + WhiteKernel(noise_level=1e-08)
best_input [0.21075737 0.08641727 0.04740185 0.2036056  0.99676354 0.39579872
 0.19577732 0.39916831]
0.210757-0.086417-0.047402-0.203606-0.996764-0.395799-0.195777-0.399168
best_output 10.004979648111663
/find a better value.
  warnings.warn(
Function 1 gp : Iter 25 | Selected UCB | Next input: [0.12422168 0.02236367] | Predicted y: 0.000199
Iter 26 | Selected UCB | Next input: [0.88061496 0.09432448] | Predicted y: 0.000075
Iter 27 | Selected UCB | Next input: [0.90661651 0.61530052] | Predicted y: -0.000150
Iter 28 | Selected UCB | Next input: [0.42761256 0.38372649] | Predicted y: -0.000048
Iter 29 | Selected UCB | Next input: [0.28212602 0.02558745] | Predicted y: 0.000253




===========================================================





Week 6 
Function 1	0.044772-0.931757
Functino 2	Best found input values: 	0.7026 -0.9266.  Input 1: 0.7026
  Input 2: 0.9266 0.702637-0.926564
Function 3 	0.482759-0.103448-0.896552 gaussian ubc 0.103448-0.103448-0.103448. best_input [0.702637 0.926564]
best_outuput 0.6840853188300118
Function 4 	0.692503-0.028572-0.244563-0.787432 used svr 0.277820-0.341278-0.509264-0.438066
Function 5 	0.241379-0.862069-0.931034-1.000000 0.701628-0.298494-0.517804-0.511965
Function 6 	[0.7281861047460138, 0.1546925696237983, 0.7325516687239811, 0.6939965090690888, 0.056401310518258585]
	0.728186-0.154693-0.732552-0.693997-0.056401 -random forest 
FUNCTION 7 	0.373252-0.854963-0.368144-0.033883-0.818517-0.927847

Function 8 	0.151419-0.139627-0.104801-0.144751-0.854834-0.514860-0.167028-0.492837
	
	
Best observed value: 0.684085

Best found input values:
  Input 1: 0.7026
  Input 2: 0.9266

Best output value: 0.6841
weekly_changes = np.diff(all_weeks_array, axis=0)

# Fraction of weeks that improved
fraction_positive = np.mean(weekly_changes > 0, axis=0)

# Average weekly change (magnitude)
avg_weekly_change = np.mean(weekly_changes, axis=0)

# Last week change
last_week_change = all_weeks_array[-1] - all_weeks_array[-2]

# Combined score: positive fraction * average weekly improvement
consistent_score = fraction_positive * avg_weekly_change

# Now pick best and worst by consistent_score
worst_funcs = np.argsort(consistent_score)[:4]
best_funcs = np.argsort(consistent_score)[-4:]

=============================================================

Function 1	Best found input: [0.98582526 0.98680981]
Best output value: 0.016626678581815213
Functino 2	Best found input: [0.99906683 0.028224  ]
Best found output: 0.5635201366379147
Function 3	0.997172-0.018252-0.743186
Function 4 	Query 15: Using existing output for [0.00637904 0.97835479 0.96574721 0.3951553 ], output = -7.2614

Best found input: [0.00637904 0.97835479 0.96574721 0.3951553 ]
Best output value: 0.7568270453454531

Best found input: [0.00637904 0.97835479 0.96574721 0.3951553 ]
Best found output: 0.7568270453454531.   Best input by confidence: [0.37216702 0.43207688 0.43940499 0.61293958]
Predicted mean: 1.0696608136424821
Function 5 	[0.27493657 0.80387933 0.95287067 0.99692827]
Output: 2375.426348037351
Function 6 	0.800599-0.033663-0.962069-0.962299-0.009327
	0.648172-0.368242-0.957155-0.140351-0.870087
FUNCTION 7 	0.025094-0.105987-0.131162-0.177208-0.382072-0.610314
Function 8 	Input: [0.11036618 0.24411988 0.30940763 0.14737193 0.90034246 0.68637932
	 0.0487734  0.63092199]
	Output: 10.0469

=====================================

    Week4 
Function 1: 	
	Best found input: [0.95671706-0.00435902]
	Best found output: 0.01672407776345608
Function 2 : 	Best found input: [0.01081871-0.996021  ]
	Best found output: 27.048564076423645
Function 3 	
Best found input: [0.96165173 0.05028473 0.09823849]
Best output value: 0.3156791893838289

Best found input: 0.96165173-0.05028473-0.09823849
Best found output: 0.3156791893838289
Function 4 	0.40623905-0.40993747-0.37736992-0.42221222
Function 5 	0.25800784-0.92580589-0.87948418-0.9576439
Function 6 	0.09702401-0.17736513-0.70421038-0.99844286-0.0129796
Function 7 	0.03702637-0.46895043-0.38643616-0.11695313-0.37409514-0.62343088
Function 8 	0.151419-0.139627-0.104801-0.144751-0.854834-0.514860-0.167028-0.492837
ut) fucntion 7 0.05111326 0.98523929 0.26563235 0.00873233 0.97481163 0.65209162
+++++------------------------------------------------------------------------------------------------------------------------------
29 oct 
Function 1 0.626692-0.224046
0.202514-0.586723
Function 2 0.56658147 0.04536399
Function 3 0.511508-0.325146-0.379547
Function 4

[0.94838936-0.89451301-0.85163782-0.55219629]

Function 5 0.374540-0.950714-0.731994-0.598658 
0.374540-0.950714-0.731994-0.598658

Fucntion 6  0.950533-0.022066-0.982521-0.086132-0.814387
Function 7 : 5.88376464e-03 4.91672219e-01 3.22680001e+00 2.96306593e+01
 4.20428330e-01 1.91432762e+02
Function 8  0.056447-0.065956-0.022929-0.038786-0.403935-0.801055-0.488307-0.893085
                     0.328954-0.834246-0.558944 0.770937-0.423857-0.281652
 

0.58637144 0.88073573 0.74502075 0.54603485 0.00964888 0.74899176
 0.23090707 0.09791562

+++++++++++++++++++++++++++++++++++++++++++++++++++++++

Function 1 : if tuple(x_next) in output_dict:
    y_next = output_dict[tuple(x_next)]
    print(f"Query {i+1}: Using existing output for {x_next}, output = {y_next:.4f}")
else:
    # If you don't have true output, you might stop or use prediction cautiously
    y_next, _ = model.predict(x_next.reshape(1, -1), return_std=True)
    y_next = y_next.item()
    print(f"Query {i+1}: No true output available; using model prediction {y_next:.4f}")

X_BO.append(x_next)
Y_BO.append(y_next)
print("\nBest found input:", X_BO[best_idx])
print("Best output value:", Y_BO[best_idx])
print("\nBest found input:", X_BO[best_idx])
print("Best found output:", Y_BO[best_idx])
Query 2: No true output available; using model prediction 0.0000

Best found input: [0.48959527 0.11687729]
Best output value: 0.0

Best found input: [0.48959527 0.11687729]
Best found output: 0.0

Function 2 : if tuple(x_next) in output_dict:
    y_next = output_dict[tuple(x_next)]
    print(f"Query {len(X_BO)+1} Using existing output for {x_next}, output = {y_next:.4f}")
else:
    # If you don't have true output, you might stop or use prediction cautiously
    y_next, _ = model.predict(x_next.reshape(1, -1), return_std=True)
    y_next = y_next.item()
    print(f"Query {len(X_BO)+1}: No true output available; using model prediction {y_next:.4f}")

    
X_BO.append(x_next)
Y_BO.append(y_next)
best_idx = np.argmax(Y_BO)

print("\nBest found input:", X_BO[best_idx])
print("Best output value:", Y_BO[best_idx])
print("\nBest found input:", X_BO[best_idx])
print("Best found output:", Y_BO[best_idx])

Best found input: [0.72378212- 0.20688659]
Best found output: 0.0
\

Function 3 Query 1: No true output available; using model prediction 0.0000

Best found input: [0.62424514 0.12208953 0.64963756]
Best output value: 0.0

Best found input: [0.62424514 0.12208953 0.64963756]
Best found output: 0.0
Function 4 
Best found input: [0.66022594 0.17494516 0.88347063 0.4348916 ]
Best output value: 0.0

Best found input: [0.66022594 0.17494516 0.88347063 0.4348916 ]
Best found output: 0.0
Function 5 : 
Best found input: [0.60814496 0.1368932  0.45419728 0.80860639]
Best found output: 0.0
fucntijon 6 Query 5: No true output available; using model prediction 0.0000

Best found input: [0.03531418 0.95836312 0.65237342 0.89866818 0.36138273]
Best output value: 0.0

Best found input: [0.03531418 0.95836312 0.65237342 0.89866818 0.36138273]
Best found output: 0.0

Function 7 
Best found input: [0.0051015  0.62827802 0.59754757 0.52190337 0.40846437 0.80482537]
Best found output: 0.0

Function 8 

Best found input: [0.12169089 0.77127108 0.03258958 0.82531454 0.04501912 0.96056743
 0.19418319 0.94131922]
Best found output: 7.231658521649857
------------------------------------------------------------
------------------->


FUNCtion 1 
0.02002565 0.99662438
Function 2
[0.70263656 0.9265642
Function 3
0.94167764 0.00638635 0.11760703
Function 4 
0.38122474 0.32878692 0.35541399 0.3971308
Function 5
0.22418902 0.84648049 0.87948418 0.87851568
Function 6 
Best found input: [0.7281861  0.15469257 0.73255167 0.69399651 0.05640131]
Best found output: -0.7142649478202404
Function 7
0.07349717 0.41118424 0.48473087 0.11720786 0.23278389 0.81487947
------------------------------------------
-----------------------------------------
Function 3 : 0.103448-0.310345-0.655172
Function 4 Next point to sample based on PI: [0.4137931  0.4137931  0.27586207 0.44827586]
0.413793-0.413793-0.275862-0.448276

Function 7 0.012457-0.520326-0.580380-0.110140-0.350434-0.713905

🔹 Processing Function 1...
✅ Function 1 - Best predicted output: 0.000000
   Best input (rounded): [0.686326 0.930263]

🔹 Processing Function 2...
✅ Function 2 - Best predicted output: -0.005288
   Best input (rounded): [0.53429  0.686038]

🔹 Processing Function 3...
✅ Function 3 - Best predicted output: -0.036355
   Best input (rounded): [0.42842  0.242703 0.529721]

🔹 Processing Function 4...
✅ Function 4 - Best predicted output: -32.625648
   Best input (rounded): [0.824189 0.977697 0.421687 0.917735]

🔹 Processing Function 5...
✅ Function 5 - Best predicted output: 437.294278
   Best input (rounded): [0.368235 0.605824 0.764554 0.663582]

🔹 Processing Function 6...
✅ Function 6 - Best predicted output: -2.545294
   Best input (rounded): [0.40639  0.022129 0.893597 0.731457 0.944492]

🔹 Processing Function 7...
✅ Function 7 - Best predicted output: 0.729186
   Best input (rounded): [0.014575 0.239636 0.801707 0.452165 0.522887 0.518328]

🔹 Processing Function 8...
✅ Function 8 - Best predicted output: 9.598482
   Best input (rounded): [0.328954 0.834246 0.558944 0.770937 0.423857 0.281652]