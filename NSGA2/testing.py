import NSGAII as ga

g = ga.random_genome(10)
print(g)
a = ga.mutation(g)
print(a)
print(g-a)

