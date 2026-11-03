import sys, numpy as np
for pid in sys.argv[1:]:
    a=dict(np.load(f".work/eqtest1/plain/pairs/pair_{pid}.npz")); b=dict(np.load(f".work/eqtest1/chunk/pairs/pair_{pid}.npz"))
    assert a.keys()==b.keys(), (set(a)^set(b))
    for k in sorted(a):
        x,y=a[k],b[k]
        if x.dtype==bool or x.dtype.kind in "iu": ok=np.array_equal(x,y); print(pid,k,"EQUAL" if ok else "DIFF")
        else: print(pid,k,"max|diff| %.2e"%np.nanmax(np.abs(x-y)) if x.size else "empty")
