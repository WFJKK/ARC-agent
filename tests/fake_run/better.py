g = graph()
xs = g.inputs(128)
e = g.entry((model.forward(g, lang, xs) == task.answer(g, lang, xs)).mean())
print(save_estimator("sampling_128", e), score_against(e, 0.98, runs=50))
