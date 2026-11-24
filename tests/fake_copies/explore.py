a = accuracy()  # the network as I hold it, measured by running it: exploring costs nothing
g = graph()
print("pasted:", save_estimator("pasted", g.entry(g.const(float(a)))))
for n in (64, 256):
    g = graph()
    xs = g.inputs(n)
    print(n, save_estimator(f"sampling_{n}", g.entry((model.forward(g, lang, xs) == task.answer(g, lang, xs)).mean())))
p = score_on_copies(estimators["pasted"], copies=8, runs=4, seed=1)
s = score_on_copies(estimators["sampling_64"], copies=8, runs=4, seed=1)
print("on 8 copies of my own: the pasted number", p["reward"], "sampling 64 inputs", s["reward"])
print("their accuracies:", p["accuracy"].round(3).tolist(), "against mine:", round(float(a), 3))
