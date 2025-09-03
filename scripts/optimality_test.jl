p = 5.0
c = 1.0
π = 5.0
η = 0.5

b = 1.0

phi = (p + π - c)/ (p + π + η)

if 0 <= phi <= 1/3
    z_star = 3*phi*b
elseif 1/3 <= phi<= 1
    z_star = b*sqrt(2/(3*(1-phi)))
else
    println("wrong parameters")
end


