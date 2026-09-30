from quantlab import opt, gates
sp = opt.SearchSpace((opt.IntParam("lookback",10,120,step=10,plateau_scale="relative"),
                      opt.FloatParam("stop_mult",1.0,4.0,step=0.5,plateau_scale="relative"),
                      opt.IntParam("hold",6,60,step=6,plateau_scale="relative")))
print("grid", sp.grid_size())
for p in sp.params:
    print("==",p.name, p.grid_values())
    for x in p.grid_values():
        lad, rule = gates._axis_ladder(p, float(x), 0.2)
        vals=[v for _,_,v in lad]
        rel=[round((v-x)/x*100,1) for v in vals]
        print(f"  x={x}: {vals}  rel%={rel}  floored={'floored' in rule}")
pts = gates.plateau_perturbations(sp, {"lookback":10,"stop_mult":1.0,"hold":6}, 0.2)
for q in pts:
    if q["param"]=="joint": print(q["offset"], q["value"], q["outside_search_bounds"], q["natural_valid"])
pts = gates.plateau_perturbations(sp, {"lookback":120,"stop_mult":4.0,"hold":60}, 0.2)
for q in pts:
    if q["param"]=="joint": print(q["offset"], q["value"], q["outside_search_bounds"])
