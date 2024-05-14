using DifferentialEquations, Plots

function f(du, u,p,t)
    mu_x = 2.8e4
    k_x = 6e-3
    kbT = 3.8
    mu_y, k_y, k_int, tau, eps = p
    D_x = kbT * mu_x
    D_y = kbT * mu_y
    du[1] = mu_x*(-k_x*u[1] + k_int*u[2])
    du[2] = mu_y*(-k_y*u[2] + k_int*u[1]) + mu_y*u[3]
    du[3] = - u[3]/tau
end

function g(du,u,p,t)
    mu_x = 2.8e4
    k_x = 6e-3
    kbT = 3.8
    mu_y, k_y, k_int, tau, eps = p
    D_x = kbT * mu_x
    D_y = kbT * mu_y
    du[1] = sqrt(2*D_x*mu_x)
    du[2] = sqrt(2*D_y*mu_y)
    du[3] = sqrt(2*eps^2 / tau)
end



function n(du,u,p,t)
    mu_x = 2.8e4
    k_x = 6e-3
    kbT = 3.8
    mu_y, k_y, k_int, tau, eps = p
    D_x = kbT * mu_x
    D_y = kbT * mu_y
    du[1] = 0 
    du[2] = 0
    du[3] = 0
end

p0 = [20e-4,7e-2,3e-3,8e-2,3]

u0 = [1.0, 1.0, 1.0]
tspan = (0.0, 1.0)

prob = SDEProblem(f, n, u0, tspan, p0)

sol = solve(prob, dt = 0.0025, adaptive = true)
plot(sol, idxs = (1))