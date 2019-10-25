def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):  
      
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 
    print ("equation of plane is ", )
    print (a, "x +", 
     b, "y +", 
     c, "z +", 
     d, "= 0.")

x1 =-1 
y1 = 2
z1 = 1
x2 = 0
y2 =-3
z2 = 2
x3 = 1
y3 = 1
z3 =-4
equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3) 
