# near-to-far-field-transformation-based-on-gprMax
Instructions:
1. When modeling in gprMax, use #python and #rx commands to place rxs around the antenna in the shape of a rectangular box.
2. Run simulation.
3. Modify main.py:<br>
a) out_path = r'the path where the .out file is located'<br> 
b) pattern_path = r'the path you want to store the far field pattern'<br> 
c) freq = 0.95e9 # Frequency points for which you want to view the pattern<br> 
d) xmin, ymin, zmin, xmax, ymax, zmax = 23,23,23,26,26,176 # The coordinates of the rectangular box, the same as in .in file<br>
4. Run main.py.
5. The far field pattern is stored in the .npz file as a 360*360 matrix.
