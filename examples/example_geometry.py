from triple_grid.geometry import Background, Color

bg = Background(2, 3, 4)

print("Vertices:")
for v in bg.vertices:
    print(v)

print()

print("R =", bg.R)
print("B =", bg.B)
print("G =", bg.G)

print()

print("Red circles")
for cid, circle in bg.cycles[Color.RED].items():
    print(cid, circle)

print()

print("Blue circles")
for cid, circle in bg.cycles[Color.BLUE].items():
    print(cid, circle)

print()

print("Green circles")
for cid, circle in bg.cycles[Color.GREEN].items():
    print(cid, circle)