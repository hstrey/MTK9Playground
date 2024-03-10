using DifferentialEquations, Distributions, ModelingToolkit, BenchmarkTools
using ModelingToolkit: t_nounits as t, D_nounits as D
# @variables t
# D = Differential(t)

mutable struct QIFNeuron
    connector::Num
    odesystem::ODESystem
    function QIFNeuron(;name, η=0.12)
        sts = @variables v(t)=-2.0 s(t)=0.0 jcn(t)=0.0
        ps = @parameters η=η
        eqs = [
            D(v) ~ v^2 + η + 15*s,
            D(s) ~ -s/0.01 + jcn,
        ]
        ev = [v ~ 200] => [v ~ -200]
        odesys = ODESystem(eqs,t,sts,ps,continuous_events=[ev];name=name)
        new(1/(1+exp(-(odesys.v - 100))), odesys)
    end
end

N = 50
neurons = []
all_sys = []
all_connects = []
for i = 1:N
    next_neuron = QIFNeuron(;name=Symbol("neuron$i"), η=rand(Cauchy(0.12, 0.02)))
    push!(neurons, next_neuron)
    push!(all_sys, next_neuron.odesystem)
    push!(all_connects, next_neuron.connector)
end

adj_matr = ones(N, N)
connection_eqs = []
for i = 1:N
    push!(connection_eqs, neurons[i].odesystem.jcn ~ sum(adj_matr[:, i] .* all_connects/N))
end

@named connection_system = ODESystem(connection_eqs, t)
@named final_sys = compose(connection_system, all_sys...)
final_sys = complete(structural_simplify(final_sys))
prob = ODEProblem(final_sys, [], (0.0, 500.0))
@btime sol = solve(prob, Tsit5(), saveat=1)