using Pkg

# Core modeling and optimization
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("MathOptInterface")

# Python interop and visualization
Pkg.add("PyCall")
Pkg.add("PyPlot")

# Image processing
Pkg.add("Images")

# Utility packages
Pkg.add("Printf")
Pkg.add("Dates")
Pkg.add("Base.Cartesian")
Pkg.add("DocStringExtensions")
Pkg.add("ProgressMeter")
Pkg.add("ArgParse")
Pkg.add("Memento")  # for logging