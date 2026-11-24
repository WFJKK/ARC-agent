g = graph()
z = g.normal(1).sum()
print(save_estimator("bad_nan", g.entry(lang.where(z > 0.0, g.const(0.0) / 0.0, z))))
