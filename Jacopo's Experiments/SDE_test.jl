using DifferentialEquations, Plots

function f(du, u,p,t)
    mu_x, mu_y, k_x, k_y, k_int, tau, D_x, D_y, eps = p
    du[1] = mu_x*(-k_x*u[1] + k_int*u[2])
    du[2] = mu_y*(-k_y*u[2] + k_int*u[1]) + mu_y*u[3]
    du[3] = - u[3]/tau
end

function g(du,u,p,t)
    mu_x, mu_y, k_x, k_y, k_int, tau, D_x, D_y, eps = p
    du[1] = sqrt(2*D_x)
    du[2] = sqrt(2*D_y)
    du[3] = sqrt(2*eps^2 / tau)
end

p0 = [1.0, 1.0 , 6.0 , 1.0, 3.0, 3.0, 2.0, 3.0, 1.0]

u0 = [1.0, 1.0, 1.0]
tspan = (0.0, 5.0)

prob = SDEProblem(f, g, u0, tspan, p0)

sol = solve(prob, dt = 0.0025, adaptive = true)
plot(sol, idxs = (1))



# ------------------------------------------------------

function lorenz(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

function σ_lorenz(du, u, p, t)
    du[1] = 3.0
    du[2] = 3.0
    du[3] = 3.0
end

prob_sde_lorenz = SDEProblem(lorenz, σ_lorenz, [1.0, 0.0, 0.0], (0.0, 10.0))
sol = solve(prob_sde_lorenz, EulerHeun(), dt = 10)
plot(sol, idxs = (1, 2, 3))



#####- ----------

function f(du, u, p, t)
    mu, tau, d, k = p
    du[1] = mu*(- (k*u[1] + d*u[1]^3) + u[2])
    du[2] = -u[2]/tau
end

function g(du, u, p, t)
    du[1] = 1.0
    du[2] = 1.0
end

u0 = [0.0, 0.0]
tspan = (0.0, 5000.0)
p = [1.0, 1.0, 0.1, 1.0]
prob = SDEProblem(f, g, u0, tspan, p)
sol = solve(prob, dt = 0.01)
plot(sol, idxs = 1)